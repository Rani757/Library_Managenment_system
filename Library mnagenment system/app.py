from flask import Flask, g, render_template_string, request, redirect, url_for, flash
import sqlite3, os
from datetime import datetime, timedelta
from groq import Groq
# register integrations (recommender, analytics, chatbot)
import integrations



app = Flask(__name__)
app.secret_key = "smart-library-key"
DB_PATH = "library.db"


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        need_init = not os.path.exists(DB_PATH)
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        if need_init:
            init_db(db)
    return db

def init_db(db):
    c = db.cursor()
    c.executescript("""
    CREATE TABLE books(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        author TEXT,
        isbn TEXT UNIQUE,
        total_copies INTEGER NOT NULL,
        available_copies INTEGER NOT NULL
    );
    CREATE TABLE members(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE
    );
    CREATE TABLE loans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER,
        member_id INTEGER,
        borrowed_at TEXT,
        due_at TEXT,
        returned_at TEXT,
        FOREIGN KEY(book_id) REFERENCES books(id),
        FOREIGN KEY(member_id) REFERENCES members(id)
    );
    """)
    db.commit()

@app.teardown_appcontext
def close_db(error):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ------------------------------------------------------------
# Base HTML Template
# ------------------------------------------------------------
BASE_HTML = """
<!doctype html>
<html>
<head>
<title>Smart Library</title>
<style>
body {font-family: Arial, sans-serif; margin: 20px; background: #f4f6fa;}
h1 {color: #2a3d66;}
.navbar {margin-bottom: 20px;}
a.button, button {
  background: #007bff; color: white; padding: 6px 12px;
  text-decoration: none; border-radius: 5px; border: none;
}
a.button:hover, button:hover {background: #0056b3;}
input, select, textarea {margin: 4px; padding: 6px;}
table {border-collapse: collapse; width: 100%; background: white;}
th, td {border: 1px solid #ddd; padding: 8px;}
th {background: #007bff; color: white;}
.success {color: green;}
.error {color: red;}
.ai-box {background:#eef3ff;padding:15px;border-radius:8px;}
</style>
</head>
<body>
<h1>📚 Smart Library Management System</h1>
<div class="navbar">
  <a class="button" href="/">Home</a>
  <a class="button" href="/books">Books</a>
  <a class="button" href="/members">Members</a>
  <a class="button" href="/loans">Loans</a>
  <a class="button" href="/ai">🤖 AI Assistant</a>
</div>

{% with msgs = get_flashed_messages() %}
{% if msgs %}
  <ul>{% for m in msgs %}<li>{{m}}</li>{% endfor %}</ul>
{% endif %}
{% endwith %}

{{ content|safe }}
</body></html>
"""


@app.route("/")
def home():
    db = get_db()
    cur = db.cursor()
    q = request.args.get("q", "").strip()

    if q:
        like = f"%{q}%"
        cur.execute("SELECT * FROM books WHERE title LIKE ? OR author LIKE ? OR isbn LIKE ?", (like, like, like))
    else:
        cur.execute("SELECT * FROM books")

    books = cur.fetchall()
    cur.execute("SELECT id,name FROM members ORDER BY name")
    members = cur.fetchall()

    html = f"""
    <h2>Search / Add Books</h2>
    <form method='get'>
      <input name='q' placeholder='Search books...' value='{q}'>
      <button type='submit'>Search</button>
    </form>

    <form method='post' action='/books/add'>
      <input name='title' placeholder='Title' required>
      <input name='author' placeholder='Author'>
      <input name='isbn' placeholder='ISBN'>
      <input type='number' name='copies' min='1' value='1'>
      <button>Add Book</button>
    </form>

    <h3>Available Books</h3>
    <table><tr><th>Title</th><th>Author</th><th>ISBN</th><th>Available</th><th>Borrow</th></tr>
    """
    for b in books:
        html += f"<tr><td>{b['title']}</td><td>{b['author'] or ''}</td><td>{b['isbn'] or ''}</td><td>{b['available_copies']} / {b['total_copies']}</td><td>"
        if b['available_copies'] > 0:
            html += f"<form method='post' action='/borrow/{b['id']}'><select name='member_id' required><option value=''>Select Member</option>"
            for m in members:
                html += f"<option value='{m['id']}'>{m['name']}</option>"
            html += "</select><input type='number' name='days' value='7' min='1' style='width:60px'><button>Borrow</button></form>"
        else:
            html += "<span class='error'>No copies</span>"
        html += "</td></tr>"
    html += "</table>"
    return render_template_string(BASE_HTML, content=html)


@app.route("/books")
def books_page():
    return redirect("/")

@app.route("/books/add", methods=["POST"])
def add_book():
    db = get_db()
    title = request.form["title"]
    author = request.form.get("author", "")
    isbn = request.form.get("isbn", "")
    copies = int(request.form.get("copies", 1))
    cur = db.cursor()
    try:
        cur.execute("INSERT INTO books(title,author,isbn,total_copies,available_copies) VALUES (?,?,?,?,?)",
                    (title, author, isbn, copies, copies))
        db.commit()
        flash("✅ Book added successfully!")
    except sqlite3.IntegrityError:
        flash("⚠️ Book with same ISBN already exists!")
    return redirect("/")

# ------------------------------------------------------------
# Members
# ----
@app.route("/members", methods=["GET", "POST"])
def members():
    db = get_db()
    cur = db.cursor()

    # Add new member
    if request.method == "POST":
        name = request.form["name"]
        email = request.form.get("email", "")
        try:
            cur.execute("INSERT INTO members(name,email) VALUES (?,?)", (name, email))
            db.commit()
            flash("✅ Member added!")
        except sqlite3.IntegrityError:
            flash("⚠️ Email already exists!")
        return redirect("/members")

    # Fetch all members
    cur.execute("SELECT * FROM members ORDER BY name")
    rows = cur.fetchall()

    # HTML for displaying members + delete button
    html = """
    <h2>Members</h2>
    <form method='post'>
      <input name='name' placeholder='Name' required>
      <input name='email' placeholder='Email'>
      <button>Add Member</button>
    </form>
    <table><tr><th>ID</th><th>Name</th><th>Email</th><th>Action</th></tr>
    """
    for r in rows:
        html += f"""
        <tr>
          <td>{r['id']}</td>
          <td>{r['name']}</td>
          <td>{r['email'] or ''}</td>
          <td>
            <form method='post' action='/members/delete/{r['id']}' onsubmit="return confirm('Delete this member?');">
              <button style='background:red;'>Delete</button>
            </form>
          </td>
        </tr>
        """
    html += "</table>"
    return render_template_string(BASE_HTML, content=html)
@app.route("/members/delete/<int:member_id>", methods=["POST"])
def delete_member(member_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM members WHERE id=?", (member_id,))
    db.commit()
    flash("🗑️ Member deleted successfully!")
    return redirect("/members")



@app.route("/borrow/<int:book_id>", methods=["POST"])
def borrow(book_id):
    db = get_db()
    cur = db.cursor()
    member_id = request.form["member_id"]
    days = int(request.form["days"])
    cur.execute("SELECT available_copies FROM books WHERE id=?", (book_id,))
    b = cur.fetchone()
    if b and b["available_copies"] > 0:
        borrowed_at = now()
        due_at = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("INSERT INTO loans(book_id,member_id,borrowed_at,due_at) VALUES (?,?,?,?)",
                    (book_id, member_id, borrowed_at, due_at))
        cur.execute("UPDATE books SET available_copies = available_copies - 1 WHERE id=?", (book_id,))
        db.commit()
        flash("✅ Book borrowed successfully!")
    else:
        flash("⚠️ Book not available!")
    return redirect("/")

@app.route("/loans")
def loans():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT loans.id,books.title,members.name,loans.borrowed_at,loans.due_at,loans.returned_at
        FROM loans
        JOIN books ON loans.book_id=books.id
        JOIN members ON loans.member_id=members.id
        ORDER BY loans.borrowed_at DESC
    """)
    rows = cur.fetchall()
    html = "<h2>Loans</h2><table><tr><th>ID</th><th>Book</th><th>Member</th><th>Borrowed</th><th>Due</th><th>Returned</th><th>Action</th></tr>"
    for r in rows:
        overdue = ""
        if not r["returned_at"] and datetime.strptime(r["due_at"], "%Y-%m-%d %H:%M:%S") < datetime.now():
            overdue = " style='color:red;font-weight:bold'"
        html += f"<tr><td>{r['id']}</td><td>{r['title']}</td><td>{r['name']}</td><td>{r['borrowed_at']}</td><td{overdue}>{r['due_at']}</td><td>{r['returned_at'] or ''}</td><td>"
        if not r["returned_at"]:
            html += f"<form method='post' action='/return/{r['id']}'><button>Return</button></form>"
        html += "</td></tr>"
    html += "</table>"
    return render_template_string(BASE_HTML, content=html)

@app.route("/return/<int:loan_id>", methods=["POST"])
def return_book(loan_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT book_id FROM loans WHERE id=? AND returned_at IS NULL", (loan_id,))
    loan = cur.fetchone()
    if loan:
        cur.execute("UPDATE loans SET returned_at=? WHERE id=?", (now(), loan_id))
        cur.execute("UPDATE books SET available_copies = available_copies + 1 WHERE id=?", (loan["book_id"],))
        db.commit()
        flash("✅ Book returned!")
    else:
        flash("⚠️ Invalid loan ID or already returned.")
    return redirect("/loans")


client = Groq(api_key="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")

@app.route("/ai", methods=["GET", "POST"])
def ai_helper():
    response, question = "", ""

    if request.method == "POST":
        question = request.form["question"]
        try:
            chat_completion = client.chat.completions.create(
                model="groq/compound-mini",
                messages=[
                    {"role": "system", "content": "You are a friendly AI assistant helping manage a library."},
                    {"role": "user", "content": question},
                ],
            )
            response = chat_completion.choices[0].message.content
        except Exception as e:
            response = f"⚠️ Error: {e}"

    html = f"""
    <div class='ai-box'>
    <h2>🤖 AI Library Assistant (Groq)</h2>
    <form method='post'>
      <textarea name='question' rows='3' cols='60' placeholder='Ask me anything about books or management...' required>{question}</textarea><br>
      <button>Ask AI</button>
    </form>
    <h3>Answer:</h3>
    <div style='white-space:pre-wrap;border:1px solid #ccc;padding:10px;border-radius:6px;background:#fff'>{response}</div>
    </div>
    """
    return render_template_string(BASE_HTML, content=html)


if __name__ == "__main__":
    print("🚀 Smart Library System running at http://127.0.0.1:5000")
    app.run(debug=True)
