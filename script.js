const wrapper = document.querySelector(".wrapper");
const loginLink = document.querySelector(".login-link");
const registerLink = document.querySelector(".register-link");

// Switch to register form
registerLink.addEventListener("click", () => {
  wrapper.classList.add("active");
  document.getElementById("login-error").textContent = "";
  document.getElementById("login-success").textContent = "";
});

// Switch to login form
loginLink.addEventListener("click", () => {
  wrapper.classList.remove("active");
  document.getElementById("register-error").textContent = "";
});

// AJAX Login
document
  .getElementById("login-form")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    const email = document.getElementById("login-email").value;
    const password = document.getElementById("login-password").value;
    const errorMsg = document.getElementById("login-error");
    const successMsg = document.getElementById("login-success");
    errorMsg.textContent = "";
    successMsg.textContent = "";

    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (res.ok && data.success) {
        document.body.classList.add("slide-up");
        setTimeout(() => {
          window.location.href = "/index";
        }, 850);
      } else {
        errorMsg.textContent = data.error || "Login failed.";
      }
    } catch (err) {
      errorMsg.textContent = "Something went wrong.";
    }
  });

// AJAX Registration
document
  .getElementById("register-form")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    const username = document.getElementById("register-username").value;
    const email = document.getElementById("register-email").value;
    const password = document.getElementById("register-password").value;
    const errorMsg = document.getElementById("register-error");
    const successMsg = document.getElementById("login-success");
    errorMsg.textContent = "";
    successMsg.textContent = "";

    try {
      const res = await fetch("/api/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, email, password }),
      });

      const data = await res.json();

      if (res.ok && data.success) {
        // Go back to login form and show success message
        wrapper.classList.remove("active");
        setTimeout(() => {
          successMsg.textContent = "✅ Registration complete! Please log in.";
        }, 300);
      } else {
        errorMsg.textContent = data.error || "Registration failed.";
      }
    } catch (err) {
      errorMsg.textContent = "Something went wrong.";
    }
  });

// Optional: Clear autofilled values on load
window.addEventListener("DOMContentLoaded", () => {
  [
    "login-email",
    "login-password",
    "register-email",
    "register-password",
  ].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
});
