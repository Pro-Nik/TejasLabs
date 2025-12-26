fetch("data/projects.json")
  .then(response => response.json())
  .then(projects => {
    const container = document.getElementById("projectsContainer");
    if (!container) return;

    projects.forEach(project => {
      const card = document.createElement("div");
      card.className = "card";

      card.innerHTML = `
        <h3>${project.title}</h3>
        <p>${project.description}</p>
        <p><strong>Role:</strong> ${project.role}</p>
        <p><strong>Technologies:</strong> ${project.technologies.join(", ")}</p>
        ${project.link ? `<a href="${project.link}" class="btn" target="_blank">View Project</a>` : ""}
      `;

      container.appendChild(card);
    });
  })
  .catch(err => console.error("Error loading projects:", err));
