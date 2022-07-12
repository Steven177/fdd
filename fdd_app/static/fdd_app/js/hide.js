function removeHidden(id) {
  var div = document.getElementById(id);
  div.classList.remove("d-none");
};

function revealHidden(id) {
  var div = document.getElementById(id);
  div.classList.remove("d-none");
};

function runFailureAnalysis(failures) {
  var div = document.getElementById("failures");
  div.classList.remove("d-none");

  document.getElementById("b6").disabled = true;
}

function hide(id) {
  var el = document.getElementById(id);
  el.classList.add("d-none");
}

