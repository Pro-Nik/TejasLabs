const container = document.getElementById("blogsContainer");

fetch("assets/data/blogs.json")
.then(res=>res.json())
.then(data=>{

    data.reverse().forEach(blog=>{

        const card=document.createElement("div");

        card.className="blog-card";

        card.innerHTML=`

        <img src="${blog.image}">

        <h2>${blog.title}</h2>

        <p>${blog.date}</p>

        <button onclick="window.location.href='blog-view.html?id=${blog.id}'">

        Read More

        </button>

        `;

        container.appendChild(card);

    });

});
