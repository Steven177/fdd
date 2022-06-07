// get references to the canvas and context
var canvas = document.getElementById("canvas");
var overlay = document.getElementById("overlay");
var ctx = canvas.getContext("2d");
var ctxo = overlay.getContext("2d");

var colors = ["green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua"];
var colorIndexExp = 0;
var colorIndexPred = 0;
var counter = 1;

// style the context
ctx.strokeStyle = colors[colorIndexExp];
ctx.lineWidth = 1.5;
ctxo.strokeStyle = colors[colorIndexExp];
ctxo.lineWidth = 1.5;

// calculate where the canvas is on the window
// (used to help calculate mouseX/mouseY)
var $canvas = $("#canvas");
var canvasOffset = $canvas.offset();
var offsetX = canvasOffset.left;
var offsetY = canvasOffset.top;
var scrollX = $canvas.scrollLeft();
var scrollY = $canvas.scrollTop();

// this flage is true when the user is dragging the mouse
var isDown = false;

// these vars will hold the starting mouse position
var startX;
var startY;

var prevStartX = 0;
var prevStartY = 0;

var prevWidth  = 0;
var prevHeight = 0;

function handleMouseDown(e) {
    e.preventDefault();
    e.stopPropagation();

    // save the starting x/y of the rectangle
    startX = parseInt(e.clientX - offsetX);
    startY = parseInt(e.clientY - offsetY);

    // set a flag indicating the drag has begun
    isDown = true;
}

function handleMouseUp(e) {
    e.preventDefault();
    e.stopPropagation();

    // the drag is over, clear the dragging flag
    isDown = false;
    ctxo.strokeRect(prevStartX, prevStartY, prevWidth, prevHeight);
    colorIndexExp += 1;
    ctx.strokeStyle = colors[colorIndexExp];
    ctxo.strokeStyle = colors[colorIndexExp];
}

function handleMouseOut(e) {
    e.preventDefault();
    e.stopPropagation();

    // the drag is over, clear the dragging flag
    isDown = false;
}

function handleMouseMove(e) {
    e.preventDefault();
    e.stopPropagation();

    // if we're not dragging, just return
    if (!isDown) {
        return;
    }

    // get the current mouse position
    mouseX = parseInt(e.clientX - offsetX);
    mouseY = parseInt(e.clientY - offsetY);

    // Put your mousemove stuff here



    // calculate the rectangle width/height based
    // on starting vs current mouse position
    var width = mouseX - startX;
    var height = mouseY - startY;

    // clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // draw a new rect from the start position
    // to the current mouse position

    ctx.strokeRect(startX, startY, width, height);

    prevStartX = startX;
    prevStartY = startY;

    prevWidth  = width;
    prevHeight = height;
}

// listen for mouse events
$("#canvas").mousedown(function (e) {
    handleMouseDown(e);
});
$("#canvas").mousemove(function (e) {
    handleMouseMove(e);
});
$("#canvas").mouseup(function (e) {
    handleMouseUp(e);
    showDiv("label" + counter);
    counter += 1;
});

$("#canvas").mouseout(function (e) {
    handleMouseOut(e);
});

// ----------------------------------------------------------------------

function revealB2() {
  b1 = document.getElementById("b1")
  b1.classList.add("d-none")

  b2 = document.getElementById("b2")
  b2.classList.remove("d-none")
}

function showModelPrediction(e) {
    revealB2();
    div = document.getElementById('hidden');
    div.style.display = "block";
};

function showDiv(div) {
  div = document.getElementById(div);
  div.style.display = "block";
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
}

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

}

$(".box").mouseover(highlightBox);
$(".box").mouseout(dehighlightBox);

