import os
from app import app, db
from models import User
from werkzeug.security import generate_password_hash

print("DB file absolute path:", os.path.abspath("wattcast.db"))

if __name__ == "__main__":
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("✅ Tables updated!")

        if not User.query.filter_by(username="admin").first():
            admin = User(
                username="admin",
                email="admin@example.com",
                password_hash=generate_password_hash("admin123")
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin user created.")


