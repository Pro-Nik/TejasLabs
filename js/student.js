function submitNotes() {
  const fileInput = document.getElementById("studentFile");

  if (!fileInput.files.length) {
    alert("Please select a file");
    return;
  }

  const submissions =
    JSON.parse(localStorage.getItem("submissions")) || [];

  submissions.push({
    student: localStorage.getItem("loggedUser"),
    file: fileInput.files[0].name,
    marks: "Pending"
  });

  localStorage.setItem("submissions", JSON.stringify(submissions));
  alert("Notes submitted successfully");
}
