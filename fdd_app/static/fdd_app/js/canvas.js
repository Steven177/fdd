// DATA
var card_raw = document.getElementById("card").innerHTML
var card = JSON.parse(card_raw)

// ------------------------------------------------

// UTILS
function create_info(text) {
  var info_tag = new fabric.Rect({
    width : 100,
    height : 80,
    stroke: 'blue',
    strokeWidth: 2,
    fill: 'rgba(0,0,0,0)',
    originX: 'center',
    originY: 'center'
  });
  var info_text = new fabric.Text(text, {
    fontSize: 30,
    originX: 'center',
    originY: 'center',
    fill: 'blue'
    })
  var info = new fabric.Group([info_tag, info_text], {
    left: 130,
    top: img_height + 60
  });
  return info;
}

function create_warning(text) {
  var tag = new fabric.Rect({
      width : 100,
      height : 80,
      stroke: "orange",
      strokeWidth: 2,
      fill: 'rgba(0,0,0,0)',
      originX: 'center',
      originY: 'center'
  });
    var text = new fabric.Text(text, {
    fontSize: 30,
    originX: 'center',
    originY: 'center',
    fill: "orange"
  })
  return [tag, text]
}

// ------------------------------------------------

// CANVAS
var canvas = new fabric.Canvas('c', {
  backgroundColor: 'white',
});


// ZOOM
canvas.on('mouse:wheel', function(opt) {
var delta = opt.e.deltaY;
var zoom = canvas.getZoom();
zoom *= 0.999 ** delta;
if (zoom > 20) zoom = 20;
if (zoom < 0.01) zoom = 0.01;
canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
opt.e.preventDefault();
opt.e.stopPropagation();
});

// ------------------------------------------------



// FAILURE CARD
for (var key in card) {
  // CARD

  // ..................................

  // EXP IMG
  var exp_img_path = document.getElementById("exp_" + key);
  var exp_img = new fabric.Image(exp_img_path,
    {left: 0,
      top: 0});

  var img_width = exp_img.width
  var img_height = exp_img.height



  if (card[key]["exp_box"] != "") {
    var exp_box = new fabric.Rect({
      left: card[key]["exp_box"]["xmin"],
      top: card[key]["exp_box"]["ymin"],
      width: card[key]["exp_box"]["xmax"] - card[key]["exp_box"]["xmin"],
      height: card[key]["exp_box"]["ymax"] - card[key]["exp_box"]["ymin"],
      stroke: "green",
      strokeWidth: 4,
      fill: 'rgba(0,0,0,0)'
    })
    var exp_vis = new fabric.Group([exp_img, exp_box])
  } else {
    var exp_vis = new fabric.Group([exp_img])
  }

  // ..................................

  // EXP LABEL
  if (card[key]["exp_label"] != "") {
    var exp_rec = new fabric.Rect({
    width : 20,
    height : 20,
    top: 5,
    left: 0,
    fill : 'green'
    });
    var exp_label = new fabric.Text(card[key]["exp_label"], {
      left: 40,
      top: 0,
      fontFamily: 'Montserrat',
      fontSize: 30
    });
    var exp = new fabric.Group([ exp_rec, exp_label ], {
      left: 10,
      top: img_height + 10,
    });
  } else {
    var exp = new fabric.Text("")

  }

  // ..................................

  // PRED IMG
  var pred_img_path = document.getElementById("pred_" + key);
  var pred_img = new fabric.Image(pred_img_path, {left: img_width + 20, top: 0});

  if (card[key]["pred_box"]  != "") {
    var pred_box = new fabric.Rect({
      left: card[key]["pred_box"]["xmin"] + img_width + 20,
      top: card[key]["pred_box"]["ymin"],
      width: card[key]["pred_box"]["xmax"] - card[key]["pred_box"]["xmin"],
      height: card[key]["pred_box"]["ymax"] - card[key]["pred_box"]["ymin"],
      stroke: "blue",
      strokeWidth: 4,
      fill: 'rgba(0,0,0,0)'
    })
    var pred_vis = new fabric.Group([pred_img, pred_box])
  } else {
    var pred_vis = new fabric.Group([pred_img])
  }

  // ..................................

  // PRED LABEL
  if (card[key]["pred_label"] != "") {
    var ped_rec = new fabric.Rect({
    width : 20,
    height : 20,
    top: 5,
    left: 0,
    fill : 'blue'
    });
    var pred_label = new fabric.Text(card[key]["pred_label"], {
      left: 40,
      top: 0,
      fontFamily: 'Montserrat',
      fontSize: 30
    });
    var pred = new fabric.Group([ ped_rec, pred_label ], {
      left: img_width + 30,
      top: img_height + 10,
    });
  } else {
    var pred = new fabric.Text("")

  }

  // ..................................


  // SEV
  var sev_circle = new fabric.Circle({
    radius: 40,
    fill: 'red',
    originX: 'center',
    originY: 'center'
  });

  var sev_text = new fabric.Text(card[key]["sev"].toString(), {
    fontSize: 30,
    originX: 'center',
    originY: 'center',
    fill: 'white'
  });

  var sev = new fabric.Group([ sev_circle, sev_text ], {
    left: img_width * 2 + 20 - 100,
    top: img_height + 60,
  });

  // ..................................


  // ERRORS
  num_of_errors = 1
  if (card[key]["tp"]) {
    var error_text = "CD"
    var error_color = "green"
  } else if (card[key]["fd"]) {
    var error_text = "FD"
    var error_color = "red"
  } else if (card[key]["md"]) {
    var error_text = "MD"
    var error_color = "red"
  } else if (card[key]["ud"]) {
    var error_text = "UD"
    var error_color = "red"
  }

  var tag = new fabric.Rect({
      width :100,
      height : 80,
      stroke: error_color,
      strokeWidth: 2,
      fill: 'rgba(0,0,0,0',
      originX: 'center',
      originY: 'center'
  });
    var text = new fabric.Text(error_text, {
    fontSize: 30,
    originX: 'center',
    originY: 'center',
    fill: error_color
  })
  var error = new fabric.Group([tag, text], {
    left: 10,
    top: img_height + 60
  })

  // ..................................


  // INFOS
  if (card[key]["id"]) {
    var ID = create_info("ID")
    var OOD = new fabric.Text("")

  }
  else if (card[key]["ood"]) {

    var OOD = create_info("OOD")
    var ID = new fabric.Text("")
  }
  else {
    var ID = new fabric.Text("")
    var OOD = new fabric.Text("")
  }

  // WARNINGS

  if (img_width * 2 + 10 < 590) {
    var shift_top = 100
    var pull_left = -240
  }
  else {
    var shift_top = 0
    var pull_left = 0
  }
  if (card[key]["ftd"] && card[key]["cqs"] && card[key]["cqb"]) {
    var FTD_s = create_warning("FTD")
    var FTD = new fabric.Group([FTD_s[0], FTD_s[1]], {
    left: 250 + pull_left,
    top: img_height + 60 + shift_top
    })
    var CQS_s = create_warning("CQS")
    var CQS = new fabric.Group([CQS_s[0], CQS_s[1]], {
    left: 370 + pull_left,
    top: img_height + 60 + shift_top
    })
    var CQB_s = create_warning("CQB")
    var CQB = new fabric.Group([CQB_s[0], CQB_s[1]], {
    left: 490 + pull_left,
    top: img_height + 60 + shift_top
    })
  }
  else if (card[key]["ftd"] && card[key]["cqs"]) {
    var FTD_s = create_warning("FTD")
    var FTD = new fabric.Group([FTD_s[0], FTD_s[1]], {
    left: 250 + pull_left,
    top: img_height + 60 + shift_top
    })
    var CQS_s = create_warning("CQS")
    var CQS = new fabric.Group([CQS_s[0], CQS_s[1]], {
    left: 370 + pull_left,
    top: img_height + 60 + shift_top
    })
    var CQB = new fabric.Text("")

  }

  else if (card[key]["cqs"] && card[key]["cqb"]) {
    var CQS_s = create_warning("CQS")
    var CQS = new fabric.Group([CQS_s[0], CQS_s[1]], {
    left: 370 + pull_left,
    top: img_height + 60 + shift_top
    })
    var CQB_s = create_warning("CQB")
    var CQB = new fabric.Group([CQB_s[0], CQB_s[1]], {
    left: 490 + pull_left,
    top: img_height + 60 + shift_top
    })
    var FTD = new fabric.Text("")
  }

  else if (card[key]["ftd"] && card[key]["cqb"]) {
    var FTD_s = create_warning("FTD")
    var FTD = new fabric.Group([FTD_s[0], FTD_s[1]], {
    left: 235 + pull_left,
    top: img_height + 60 + shift_top
    })
    var CQB_s = create_warning("CQB")
    var CQB = new fabric.Group([CQB_s[0], CQB_s[1]], {
    left: 490 + pull_left,
    top: img_height + 60 + shift_top
    })
    var CQS = new fabric.Text("")
  }

  else if (card[key]["ftd"]) {
    var FTD_s = create_warning("FTD")
    var FTD = new fabric.Group([FTD_s[0], FTD_s[1]], {
    left: 250 + pull_left,
    top: img_height + 60 + shift_top
    })
    var CQS = new fabric.Text("")
    var CQB = new fabric.Text("")
  }

  else if (card[key]["cqs"]) {
    var CQS_s = create_warning("CQS")
    var CQS = new fabric.Group([CQS_s[0], CQS_s[1]], {
    left: 370 + pull_left,
    top: img_height + 60 + shift_top
    })
    var FTD = new fabric.Text("")
    var CQB = new fabric.Text("")
  }

  else if (card[key]["cqb"]) {
    var CQB_s = create_warning("CQB")
    var CQB = new fabric.Group([CQB_s[0], CQB_s[1]], {
    left: 490 + pull_left,
    top: img_height + 60 + shift_top
    })
    var FTD = new fabric.Text("")
    var CQS = new fabric.Text("")
  }

  else {
    var FTD = new fabric.Text("")
    var CQS = new fabric.Text("")
    var CQB = new fabric.Text("")
  }

  var grey_card = new fabric.Rect( {
    width: img_width * 2 + 20,
    height: img_height + 60 + 80 + 40 + shift_top - 20,
    fill: "#F9FAFB",
    top: 0,
    left: 0
  })

  var failure_card = new fabric.Group([grey_card, exp_vis, pred_vis, sev, exp, pred, error, ID, OOD, FTD, CQS, CQB], {
  })

  canvas.add(failure_card)
}

// --------------------------------------------------------------------
// --------------------------------------------------------------------

var d = function(id){return document.getElementById(id)};
var group = d('group')
var ungroup = d('ungroup')
var add_group_name = d('add_group_name')
var add_rec = d('add_rec')
var delete_item = d('delete_item')
var save_canvas = d('save_canvas')


group.onclick = function() {
  if (!canvas.getActiveObject()) {
    return;
  }
  if (canvas.getActiveObject().type !== 'activeSelection') {
    return;
  }
  canvas.getActiveObject().toGroup();
  var active_objects = canvas.getActiveObject()
  canvas.add(group_name)
  canvas.requestRenderAll();
}


 ungroup.onclick = function() {
  if (!canvas.getActiveObject()) {
    return;
  }
  if (canvas.getActiveObject().type !== 'group') {
    return;
  }
  canvas.getActiveObject().toActiveSelection();
  canvas.requestRenderAll();
}

add_group_name.onclick = function() {
  var group_name = new fabric.IText("Name group ...", {
    fontFamily: 'Montserrat',
    fontSize: 50,
    fill: "#7207B7",
  });
  canvas.add(group_name)
}

add_rec.onclick = function() {
  var recoveries = ["E.g. communicate quality of output", " E.g. show N-best options", "E.g. hand-over control to the user", "E.g. leverage implicit feedback", "E.g. request explicit feedback", "E.g. let user make correction", "E.g. provide local explanation", "E.g. provide global explanantion"]
  var random_index = Math.floor(Math.random(7) * 10)
  var recovery = new fabric.IText(recoveries[random_index], {
    fontFamily: 'Montserrat',
    fontSize: 50,
    fill: "#ECA355",
    originX: "left",
    originY: "top",
    top: 0,
    left: 0

  });
  canvas.add(recovery)
}

delete_item.onclick = function() {
  canvas.remove(canvas.getActiveObject());
}

download_img = function(el) {
  var image = canvas.toDataURL("image/jpg");
  el.href = image;
};




