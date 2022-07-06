var counter = 0;
var colors = ["green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua"];
var dollar = document.querySelector.bind(document);

/**
 * Collection of rectangles defining user generated regions
 */
var rectangles = [];

// DOM elements
const $screenshot = dollar('#screenshot');
const $draw = dollar('#draw');
const $marquee = dollar('#marquee');
const $boxes = dollar('#boxes');

// Temp variables
let startX = 0;
let startY = 0;
const marqueeRect = {
  x: 0,
  y: 0,
  width: 0,
  height: 0,
};

$marquee.classList.add('hide');
$screenshot.addEventListener('pointerdown', startDrag);

function startDrag(ev) {
  // middle button delete rect
  if (ev.button === 1) {
    const rect = hitTest(ev.layerX, ev.layerY);
    if (rect) {
      rectangles.splice(rectangles.indexOf(rect), 1);
      redraw();
    }
    return;
  }
  window.addEventListener('pointerup', stopDrag);
  $screenshot.addEventListener('pointermove', moveDrag);
  $marquee.classList.remove('hide');
  startX = ev.layerX;
  startY = ev.layerY;
  drawRect($marquee, startX, startY, 0, 0);
}

function stopDrag(ev) {
  $marquee.classList.add('hide');
  window.removeEventListener('pointerup', stopDrag);
  $screenshot.removeEventListener('pointermove', moveDrag);
  if (ev.target === $screenshot && marqueeRect.width && marqueeRect.height) {
    rectangles.push(Object.assign({}, marqueeRect));
    redraw();
  }
  removeHidden("label" + counter);
  counter += 1;
}

function moveDrag(ev) {
  let x = ev.layerX;
  let y = ev.layerY;
  let width = startX - x;
  let height = startY - y;
  if (width < 0) {
    width *= -1;
    x -= width;
  }
  if (height < 0) {
    height *= -1;
    y -= height;
  }
  Object.assign(marqueeRect, { x, y, width, height });
  drawRect($marquee, marqueeRect);
}

function hitTest(x, y) {
  return rectangles.find(rect => (
    x >= rect.x &&
    y >= rect.y &&
    x <= rect.x + rect.width &&
    y <= rect.y + rect.height
  ));
}

function redraw() {
  boxes.innerHTML = '';
  var colorCounter = 0;
  rectangles.forEach((data) => {
    var newRect = drawRect(document.createElementNS("http://www.w3.org/2000/svg", 'rect'), data);
    newRect.style.stroke = colors[colorCounter];
    boxes.appendChild(newRect);
    colorCounter += 1;
  });
}

function drawRect(rect, data) {
  const { x, y, width, height } = data;
  rect.setAttributeNS(null, 'width', width);
  rect.setAttributeNS(null, 'height', height);
  rect.setAttributeNS(null, 'x', x);
  rect.setAttributeNS(null, 'y', y);
  return rect;
}

function clearCanvas() {
  // clear the canvas
  rectangles = [];
  redraw();
  labels = document.querySelectorAll(".label");
  for (let i=0; i<labels.length; i++) {
    label = labels[i];
    label.classList.add("d-none");
  }
  counter = 0;
};


// ----------------------------------------------------------------------


function removeHidden(id) {
  var div = document.getElementById(id);
  div.classList.remove("d-none");
};

function revealHidden(id) {
  var div = document.getElementById(id);
  div.classList.remove("d-none");
};

function showModelPrediction(id) {
  // model prediction
  var div = document.getElementById(id);
  div.style.display = "block";

  // hide button
  document.getElementById("b3").disabled = true;

  // Show failure section
  revealHidden("hidden2");
};

function highlightBox(e) {
  // Get elements
  var id = e.target.id
  var highlightBox = document.getElementById(id.toString());
  var predID = parseInt(id) + 100
  var highlightPred = document.getElementById(predID.toString())

  boxes = document.querySelectorAll(".box")
  predictions = document.querySelectorAll(".pred")

  // boxes
  for (let i = 0; i < boxes.length; i++) {
    element = boxes[i];
    element.classList.remove("normalBox");
    element.classList.add("removeBox");
  }
  highlightBox.classList.remove("removeBox")
  highlightBox.classList.add("normalBox")

  // predictions
  for (let i = 0; i < predictions.length; i++) {
    element = predictions[i];
    element.classList.add("hideDiv");
  }
  highlightPred.classList.remove("hideDiv");
};

function dehighlightBox(e) {
  // boxes
  boxes = document.querySelectorAll(".box")
  for (let i = 0; i < boxes.length; i++) {
    element = boxes[i]
    element.classList.remove("removeBox");
    element.classList.add("normalBox");
  }

  // predictions
  predictions = document.querySelectorAll(".pred")
  for (let i = 0; i < predictions.length; i++) {
    pred = predictions[i]
    pred.classList.remove("hideDiv");
  }
};

var bboxes = document.querySelectorAll('.box');
for (let i = 0; i < bboxes.length; i++) {
  bboxes[i].addEventListener("mouseover", highlightBox);
  bboxes[i].addEventListener("mouseout", dehighlightBox);
};


// ----------------------------------------------------------------------

// https://stackoverflow.com/questions/25862798/how-to-send-the-javascript-list-of-dictionaries-object-to-django-ajax
function saveBoxes() {

  var expBoxes = JSON.stringify(rectangles);
  console.log(expBoxes);
  $.ajax({
    "url": "/fdd_app/user",
    "type": "POST",
    "data": {'expBoxes[]': expBoxes},
 });

  //var b2 = document.getElementById("b2");
  //b2.classList.add("d-none");
  //var b4 document.getElementById("b4");
  //b4.classList.remove("d-none");
}
