fetch("data/blogs.json")
  .then(res => res.json())
  .then(blogs => {
    const container = document.getElementById("blogContainer");

    blogs.forEach(blog => {
      const article = document.createElement("article");
      article.className = "blog-card";
      article.innerHTML = `
        <h3>${blog.title}</h3>
        <small>${blog.author} • ${blog.date}</small>
        <div class="blog-content">
          ${renderMarkdown(blog.content)}
        </div>
      `;
      container.appendChild(article);
    });
  });

function renderMarkdown(text) {
  return text
    .replace(/^### (.*$)/gim, "<h4>$1</h4>")
    .replace(/^## (.*$)/gim, "<h3>$1</h3>")
    .replace(/^# (.*$)/gim, "<h2>$1</h2>")
    .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
    .replace(/```([\s\S]*?)```/gim, "<pre><code>$1</code></pre>")
    .replace(/\n/gim, "<br>");
}
