const API_KEY = "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";

const SYSTEM_PROMPT = `
You are an Assistant Professor in Computer Science.
Your tone is academic, supportive, clear, and motivating.
Explain concepts like an educator.
Encourage learning and critical thinking.
Avoid casual slang.
`;

async function sendMessage() {
  const input = document.getElementById("userInput");
  const chat = document.getElementById("chatWindow");

  if (!input.value.trim()) return;

  // Show user message
  chat.innerHTML += `<div class="msg user">${input.value}</div>`;
  chat.scrollTop = chat.scrollHeight;

  const userText = input.value;
  input.value = "";

  // Show typing
  chat.innerHTML += `<div class="msg bot">Thinking like a professor...</div>`;
  chat.scrollTop = chat.scrollHeight;

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${API_KEY}`
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userText }
      ]
    })
  });

  const data = await response.json();

  // Remove typing
  chat.lastChild.remove();

  const reply = data.choices[0].message.content;

  chat.innerHTML += `<div class="msg bot">${reply}</div>`;
  chat.scrollTop = chat.scrollHeight;
}
