const handleButtonClick = () => {
  let placeholder = document.getElementById("placeholder");
  placeholder.innerHTML = "";
  let errorPlaceholder = document.getElementById("error");
  errorPlaceholder.innerHTML = "";
  generateHandWriting();
};

const generateHandWriting = () => {
  const inputValue = document.getElementById("textInput").value;

  if (!inputValue) {
    appendError("Please introduce something in the input.");
    return;
  }

  const baseurl = "http://127.0.0.1:8000";
  const path = `${baseurl}/generate/mnist/${inputValue}`;

  fetch(path)
    .then((response) => {
      if (!response.ok) {
        response.text().then(text => {
          appendError(JSON.parse(text)?.detail);
        })
      } else {
        response.text().then(text => {
          appendImage(JSON.parse(text)?.path);
        })
      }
    })
};

const appendImage = (src) => {
  let img = document.createElement("img");
  img.src = src;
  const placeholder = document.getElementById("placeholder");
  placeholder.appendChild(img);
};

const appendError = (errorMessage) => {
  let text = document.createElement("p");
  text.innerText = errorMessage ? errorMessage : "Error, something went wrong. Please try again.";
  const errorPlaceholder = document.getElementById("error");
  errorPlaceholder.appendChild(text);
};

document.getElementById("generateButton").addEventListener("click", handleButtonClick);
