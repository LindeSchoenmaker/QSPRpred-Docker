from app import app, db
from app.models import User

app.app_context().push()

if __name__ == "__main__":
    u = User(username='testuser', email='test@example.com')
    u.set_password('mypassword')
    db.session.add(u)
    db.session.commit()