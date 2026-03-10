const API_BASE_URL = "http://127.0.0.1:8000";

const pdfFileInput = document.getElementById("pdfFile");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

const filenameInput = document.getElementById("filenameInput");
const chunkSizeInput = document.getElementById("chunkSize");
const chunkOverlapInput = document.getElementById("chunkOverlap");
const indexBtn = document.getElementById("indexBtn");
const indexStatus = document.getElementById("indexStatus");

const questionInput = document.getElementById("questionInput");
const topKInput = document.getElementById("topKInput");
const askBtn = document.getElementById("askBtn");
const askStatus = document.getElementById("askStatus");
const answerOutput = document.getElementById("answerOutput");
const sourcesOutput = document.getElementById("sourcesOutput");
const chunksOutput = document.getElementById("chunksOutput");

function setText(element, text) {
  element.textContent = text;
}

uploadBtn.addEventListener("click", async () => {
  const file = pdfFileInput.files[0];

  if (!file) {
    setText(uploadStatus, "Please choose a PDF file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  setText(uploadStatus, "Uploading...");

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Upload failed.");
    }

    filenameInput.value = data.filename;
    setText(
      uploadStatus,
      `Upload successful.\nFilename: ${data.filename}\nSaved path: ${data.saved_path}`
    );
  } catch (error) {
    setText(uploadStatus, `Upload failed: ${error.message}`);
  }
});

indexBtn.addEventListener("click", async () => {
  const filename = filenameInput.value.trim();
  const chunkSize = Number(chunkSizeInput.value);
  const chunkOverlap = Number(chunkOverlapInput.value);

  if (!filename) {
    setText(indexStatus, "Please enter or upload a filename first.");
    return;
  }

  setText(indexStatus, "Indexing document...");

  try {
    const url = new URL(`${API_BASE_URL}/index/${encodeURIComponent(filename)}`);
    url.searchParams.set("chunk_size", String(chunkSize));
    url.searchParams.set("chunk_overlap", String(chunkOverlap));

    const response = await fetch(url, {
      method: "POST",
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Indexing failed.");
    }

    setText(
      indexStatus,
      `Index successful.\nChunk count: ${data.chunk_count}\nInserted count: ${data.inserted_count}\nCollection: ${data.collection_name}`
    );
  } catch (error) {
    setText(indexStatus, `Index failed: ${error.message}`);
  }
});

askBtn.addEventListener("click", async () => {
  const filename = filenameInput.value.trim();
  const question = questionInput.value.trim();
  const topK = Number(topKInput.value);

  if (!question) {
    setText(askStatus, "Please enter a question.");
    return;
  }

  setText(askStatus, "Asking question...");
  setText(answerOutput, "");
  setText(sourcesOutput, "");
  setText(chunksOutput, "");

  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k: topK,
        filename,
        }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Ask request failed.");
    }

    setText(askStatus, "Question answered successfully.");
    setText(answerOutput, data.answer || "No answer returned.");
    setText(sourcesOutput, JSON.stringify(data.sources, null, 2));
    setText(chunksOutput, JSON.stringify(data.retrieved_chunks, null, 2));
  } catch (error) {
    setText(askStatus, `Ask failed: ${error.message}`);
  }
});