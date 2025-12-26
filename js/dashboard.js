function loadDashboard() {
  loadUser();
  loadBlogs();
  loadAnnouncements();
  loadNotes();
  loadProjects();
  loadMessages();
}

/* BLOGS */
function saveBlog() {
  const text = document.getElementById("blogText").value;
  if (!text) return;
  const blogs = JSON.parse(localStorage.getItem("blogs")) || [];
  blogs.push(text);
  localStorage.setItem("blogs", JSON.stringify(blogs));
  loadBlogs();
}

function loadBlogs() {
  const blogs = JSON.parse(localStorage.getItem("blogs")) || [];
  const div = document.getElementById("blogList");
  div.innerHTML = blogs.map(b => `<p>📝 ${b}</p>`).join("");
}

/* ANNOUNCEMENTS */
function addAnnouncement() {
  const text = document.getElementById("announcementText").value;
  const list = JSON.parse(localStorage.getItem("announcements")) || [];
  list.push(text);
  localStorage.setItem("announcements", JSON.stringify(list));
  loadAnnouncements();
}

function loadAnnouncements() {
  const list = JSON.parse(localStorage.getItem("announcements")) || [];
  document.getElementById("announcementList").innerHTML =
    list.map(a => `<li>${a}</li>`).join("");
}

/* NOTES */
function uploadNote() {
  const name = document.getElementById("studentName").value;
  const file = document.getElementById("noteFile").files[0];
  if (!name || !file) return;
  const notes = JSON.parse(localStorage.getItem("notes")) || [];
  notes.push({ name, file: file.name, marks: "Pending" });
  localStorage.setItem("notes", JSON.stringify(notes));
  loadNotes();
}

function loadNotes() {
  const notes = JSON.parse(localStorage.getItem("notes")) || [];
  document.getElementById("notesList").innerHTML =
    notes.map(n =>
      `<li>${n.name} – ${n.file} | Marks: ${n.marks}</li>`
    ).join("");
}

/* PROJECTS */
function addProject() {
  const title = document.getElementById("projectTitle").value;
  const desc = document.getElementById("projectDesc").value;
  const projects = JSON.parse(localStorage.getItem("projects")) || [];
  projects.push({ title, desc });
  localStorage.setItem("projects", JSON.stringify(projects));
  loadProjects();
}

function loadProjects() {
  const projects = JSON.parse(localStorage.getItem("projects")) || [];
  document.getElementById("projectList").innerHTML =
    projects.map(p => `<li><strong>${p.title}</strong>: ${p.desc}</li>`).join("");
}

/* MESSAGES */
function sendMessage() {
  const name = document.getElementById("msgName").value;
  const msg = document.getElementById("msgText").value;
  const msgs = JSON.parse(localStorage.getItem("messages")) || [];
  msgs.push({ name, msg });
  localStorage.setItem("messages", JSON.stringify(msgs));
  loadMessages();
}

function loadMessages() {
  const msgs = JSON.parse(localStorage.getItem("messages")) || [];
  document.getElementById("messageList").innerHTML =
    msgs.map(m => `<li><strong>${m.name}:</strong> ${m.msg}</li>`).join("");
}
