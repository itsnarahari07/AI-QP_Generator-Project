"""
AI-Based Question Paper Generator
AI  : Google Gemini 2.0 Flash Lite  (free)
SDK : google-genai   (pip install google-genai)
Run : python server.py
"""

import os, json, re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pymysql, pymysql.cursors
from dotenv import load_dotenv
import PyPDF2
from groq import Groq

# ── env ──────────────────────────────────────
load_dotenv()
KEY = os.getenv("GROQ_API_KEY", "")
if KEY:
    print(f"Groq key loaded: {KEY[:20]}...")
else:
    print("GROQ_API_KEY missing in .env")

client = Groq(api_key=KEY)
MODEL  = "llama-3.3-70b-versatile"   # free on Groq

# ── Flask ────────────────────────────────────
app = Flask(__name__)
CORS(app)
UPLOADS = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOADS, exist_ok=True)
app.config["UPLOAD_FOLDER"]      = UPLOADS
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# ── DB ───────────────────────────────────────
def db():
    return pymysql.connect(
        host=os.getenv("DB_HOST","localhost"),
        user=os.getenv("DB_USER","root"),
        password=os.getenv("DB_PASSWORD","narahari07"),
        database=os.getenv("DB_NAME","question_paper_db"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )

def init_db():
    c = db()
    try:
        with c.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS uploaded_pdfs(
                id INT AUTO_INCREMENT PRIMARY KEY,
                original_filename VARCHAR(255) NOT NULL,
                stored_filename VARCHAR(255) NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                file_size_kb DECIMAL(10,2),
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status ENUM('uploaded','processing','completed','failed') DEFAULT 'uploaded'
            ) CHARACTER SET utf8mb4""")
            cur.execute("""CREATE TABLE IF NOT EXISTS question_papers(
                id INT AUTO_INCREMENT PRIMARY KEY,
                pdf_id INT NOT NULL,
                paper_title VARCHAR(300),
                subject_name VARCHAR(200),
                unit_name VARCHAR(200),
                total_marks INT,
                generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(pdf_id) REFERENCES uploaded_pdfs(id) ON DELETE CASCADE
            ) CHARACTER SET utf8mb4""")
            cur.execute("""CREATE TABLE IF NOT EXISTS questions(
                id INT AUTO_INCREMENT PRIMARY KEY,
                paper_id INT NOT NULL,
                question_text TEXT NOT NULL,
                marks INT NOT NULL,
                question_type ENUM('short','medium','long') NOT NULL,
                unit_reference VARCHAR(200),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(paper_id) REFERENCES question_papers(id) ON DELETE CASCADE
            ) CHARACTER SET utf8mb4""")
        c.commit()
        print("Database tables ready.")
    except Exception as e:
        print(f"DB init error: {e}")
    finally:
        c.close()

# ── helpers ──────────────────────────────────
def allowed(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() == "pdf"

def read_pdf(path):
    txt = ""
    with open(path,"rb") as f:
        for pg in PyPDF2.PdfReader(f).pages:
            t = pg.extract_text()
            if t: txt += t + "\n"
    return txt.strip()

def make_prompt(subject, unit, n2, n5, n10, content):
    return (
        "You are an expert exam question paper generator.\n"
        "Read the study material and generate questions.\n\n"
        f"Subject: {subject}\nUnit: {unit}\n\n"
        f"Generate EXACTLY {n2} questions worth 2 marks, "
        f"{n5} questions worth 5 marks, "
        f"{n10} questions worth 10 marks.\n\n"
        "Return ONLY a JSON array. No extra text. No markdown fences.\n"
        "Format:\n"
        '[{"question":"...","marks":2,"type":"short"},'
        '{"question":"...","marks":5,"type":"medium"},'
        '{"question":"...","marks":10,"type":"long"}]\n\n'
        f"Study Material:\n{content[:4000]}"
    )

def ask_gemini(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=2000,
    )
    raw = resp.choices[0].message.content.strip()
    print("Groq preview:", raw[:200])
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw).strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        raise ValueError(f"No JSON array from Groq: {raw[:200]}")
    qs = json.loads(m.group())
    if not qs:
        raise ValueError("Empty question list from Groq")
    return qs

# ── routes ───────────────────────────────────
@app.route("/")
def health():
    return jsonify({"status":"running","model":MODEL,"ai":"Groq Llama 3.3 70B (free)"})

@app.route("/api/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error":"Use key 'pdf'"}), 400
    f = request.files["pdf"]
    if not f or not f.filename:
        return jsonify({"error":"No file"}), 400
    if not allowed(f.filename):
        return jsonify({"error":"PDF only"}), 400

    orig  = f.filename
    safe  = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(orig)
    path  = os.path.join(UPLOADS, safe)
    f.save(path)
    kb = round(os.path.getsize(path)/1024, 2)

    c = db()
    try:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO uploaded_pdfs(original_filename,stored_filename,file_path,file_size_kb,status) VALUES(%s,%s,%s,%s,'uploaded')",
                (orig, safe, path, kb)
            )
            pid = cur.lastrowid
        c.commit()
        print(f"Upload OK — pdf_id={pid} file={orig}")
        return jsonify({"success":True,"pdf_id":pid,"filename":orig,"size_kb":kb}), 201
    except Exception as e:
        c.rollback()
        return jsonify({"error":str(e)}), 500
    finally:
        c.close()

@app.route("/api/upload/history")
def upload_history():
    c = db()
    try:
        with c.cursor() as cur:
            cur.execute("SELECT * FROM uploaded_pdfs ORDER BY upload_timestamp DESC LIMIT 20")
            return jsonify(cur.fetchall()), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    finally:
        c.close()

@app.route("/api/generate", methods=["POST"])
def generate():
    d = request.get_json(force=True)
    pdf_id  = d.get("pdf_id")
    subject = d.get("subject_name","").strip()
    unit    = d.get("unit_name","").strip()
    n2      = int(d.get("num_2mark",5))
    n5      = int(d.get("num_5mark",3))
    n10     = int(d.get("num_10mark",2))

    if not pdf_id:  return jsonify({"error":"pdf_id required"}), 400
    if not subject: return jsonify({"error":"subject_name required"}), 400
    if not unit:    return jsonify({"error":"unit_name required"}), 400

    c = db()
    try:
        with c.cursor() as cur:
            cur.execute("SELECT * FROM uploaded_pdfs WHERE id=%s",(pdf_id,))
            rec = cur.fetchone()
        if not rec:
            return jsonify({"error":f"PDF {pdf_id} not found"}), 404

        with c.cursor() as cur:
            cur.execute("UPDATE uploaded_pdfs SET status='processing' WHERE id=%s",(pdf_id,))
        c.commit()

        txt = read_pdf(rec["file_path"])
        if not txt:
            return jsonify({"error":"Cannot extract text from PDF"}), 422
        print(f"Extracted {len(txt)} chars")

        prompt = make_prompt(subject, unit, n2, n5, n10, txt)
        qs     = ask_gemini(prompt)
        print(f"Got {len(qs)} questions")

        total  = n2*2 + n5*5 + n10*10
        title  = f"{subject} - {unit}"

        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO question_papers(pdf_id,paper_title,subject_name,unit_name,total_marks) VALUES(%s,%s,%s,%s,%s)",
                (pdf_id, title, subject, unit, total)
            )
            paper_id = cur.lastrowid
        c.commit()

        tmap = {2:"short",5:"medium",10:"long"}
        with c.cursor() as cur:
            for q in qs:
                mk = int(q.get("marks",2))
                cur.execute(
                    "INSERT INTO questions(paper_id,question_text,marks,question_type,unit_reference) VALUES(%s,%s,%s,%s,%s)",
                    (paper_id, q["question"], mk, q.get("type",tmap.get(mk,"short")), unit)
                )
        c.commit()

        with c.cursor() as cur:
            cur.execute("UPDATE uploaded_pdfs SET status='completed' WHERE id=%s",(pdf_id,))
        c.commit()

        print(f"Paper saved — paper_id={paper_id} marks={total}")
        return jsonify({
            "success":True,"paper_id":paper_id,
            "subject":subject,"unit":unit,
            "total_marks":total,"questions":qs
        }), 201

    except Exception as e:
        c.rollback()
        try:
            with c.cursor() as cur:
                cur.execute("UPDATE uploaded_pdfs SET status='failed' WHERE id=%s",(pdf_id,))
            c.commit()
        except: pass
        print(f"Generate error: {e}")
        return jsonify({"error":str(e)}), 500
    finally:
        c.close()

@app.route("/api/generate/history")
def gen_history():
    c = db()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT qp.*,up.original_filename FROM question_papers qp "
                "JOIN uploaded_pdfs up ON qp.pdf_id=up.id "
                "ORDER BY qp.generated_at DESC LIMIT 20"
            )
            return jsonify(cur.fetchall()), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    finally:
        c.close()

@app.route("/api/generate/<int:pid>")
def get_paper(pid):
    c = db()
    try:
        with c.cursor() as cur:
            cur.execute("SELECT * FROM question_papers WHERE id=%s",(pid,))
            paper = cur.fetchone()
        if not paper:
            return jsonify({"error":"Not found"}), 404
        with c.cursor() as cur:
            cur.execute("SELECT * FROM questions WHERE paper_id=%s ORDER BY marks",(pid,))
            qs = cur.fetchall()
        return jsonify({"paper":paper,"questions":qs}), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    finally:
        c.close()

@app.route("/api/generate/<int:pid>", methods=["DELETE"])
def del_paper(pid):
    c = db()
    try:
        with c.cursor() as cur:
            cur.execute("DELETE FROM question_papers WHERE id=%s",(pid,))
        c.commit()
        return jsonify({"success":True}), 200
    except Exception as e:
        c.rollback()
        return jsonify({"error":str(e)}), 500
    finally:
        c.close()

# ── run ──────────────────────────────────────
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT",5000))
    print(f"Server running → http://localhost:{port}")
    app.run(debug=True, port=port)