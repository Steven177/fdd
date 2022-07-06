function loadExplanation(id) {

  description = document.getElementById(parseInt(id) + 100);
  explanation = document.getElementById(id);
  description.classList.remove("d-none");
  explanation.src = `/media/explanations/explanation${id}.png`
  // hide button
  document.getElementById("b5").disabled = true;
}
