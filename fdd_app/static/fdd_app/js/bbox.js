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

c0.addEventListener('keyup', function(e){
  document.getElementById("x0").value = rectangles['0']['x']
  document.getElementById("y0").value = rectangles['0']['y']
  document.getElementById("width0").value = rectangles['0']['width']
  document.getElementById("height0").value = rectangles['0']['height']
})

c1.addEventListener('keyup', function(e){
  document.getElementById("x1").value = rectangles['1']['x']
  document.getElementById("y1").value = rectangles['1']['y']
  document.getElementById("width1").value = rectangles['1']['width']
  document.getElementById("height1").value = rectangles['1']['height']
})

c2.addEventListener('keyup', function(e){
  document.getElementById("x2").value = rectangles['2']['x']
  document.getElementById("y2").value = rectangles['2']['y']
  document.getElementById("width2").value = rectangles['2']['width']
  document.getElementById("height2").value = rectangles['2']['height']
})

c3.addEventListener('keyup', function(e){
  document.getElementById("x3").value = rectangles['3']['x']
  document.getElementById("y3").value = rectangles['3']['y']
  document.getElementById("width3").value = rectangles['3']['width']
  document.getElementById("height3").value = rectangles['3']['height']
})

c4.addEventListener('keyup', function(e){
  document.getElementById("x4").value = rectangles['4']['x']
  document.getElementById("y4").value = rectangles['4']['y']
  document.getElementById("width4").value = rectangles['4']['width']
  document.getElementById("height4").value = rectangles['4']['height']
})

c5.addEventListener('keyup', function(e){
  document.getElementById("x5").value = rectangles['5']['x']
  document.getElementById("y5").value = rectangles['5']['y']
  document.getElementById("width5").value = rectangles['5']['width']
  document.getElementById("height5").value = rectangles['5']['height']
})
c6.addEventListener('keyup', function(e){
  document.getElementById("x6").value = rectangles['6']['x']
  document.getElementById("y6").value = rectangles['6']['y']
  document.getElementById("width6").value = rectangles['6']['width']
  document.getElementById("height6").value = rectangles['6']['height']
})
c7.addEventListener('keyup', function(e){
  document.getElementById("x7").value = rectangles['4']['x']
  document.getElementById("y7").value = rectangles['4']['y']
  document.getElementById("width7").value = rectangles['7']['width']
  document.getElementById("height7").value = rectangles['7']['height']
})
c8.addEventListener('keyup', function(e){
  document.getElementById("x8").value = rectangles['8']['x']
  document.getElementById("y8").value = rectangles['8']['y']
  document.getElementById("width8").value = rectangles['8']['width']
  document.getElementById("height8").value = rectangles['8']['height']
})
c9.addEventListener('keyup', function(e){
  document.getElementById("x9").value = rectangles['9']['x']
  document.getElementById("y9").value = rectangles['9']['y']
  document.getElementById("width9").value = rectangles['9']['width']
  document.getElementById("height9").value = rectangles['9']['height']
})

