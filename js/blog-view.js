const id=new URLSearchParams(window.location.search).get("id");

fetch("assets/data/blogs.json")

.then(res=>res.json())

.then(data=>{

const blog=data.find(b=>b.id==id);

document.getElementById("title").innerText=blog.title;

document.getElementById("date").innerText=blog.date;

document.getElementById("image").src=blog.image;

fetch(blog.file)

.then(r=>r.text())

.then(text=>{

document.getElementById("content").innerText=text;

});

});
