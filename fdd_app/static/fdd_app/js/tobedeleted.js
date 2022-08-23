// ungroup objects in group
var items
var ungrouping = function (group) {
    items = group._objects;
    group._restoreObjectsState();
    canvas.remove(group);
    for (var i = 0; i < items.length; i++) {
        canvas.add(items[i]);
    }
    // if you have disabled render on addition
    canvas.renderAll();
};

// Re-group when text editing finishes
var group_name = new fabric.IText("Name group", {
    fontFamily: 'Montserrat',
    fontSize: 14,
    fill: "#7207B7",
    left: 170,
    top: 60
});
group_name.on('editing:exited', function () {
    var items = [];
    canvas.forEachObject(function (obj) {
        items.push(obj);
        canvas.remove(obj);
    });
    var grp = new fabric.Group(items.reverse(), {});
    canvas.add(grp);
    grp.on('mousedown', fabricDblClick(grp, function (obj) {
        ungrouping(grp);
        canvas.setActiveObject(group_name);
        group_name.enterEditing();
        group_name.selectAll();
    }));
});

function addRuler() {
    var dimension_mark = new fabric.Path('M0,0L0,-5L0,5L0,0L150,0L150,-5L150,5L150,0z');
    dimension_mark.set({
        left: 150,
        top: 70,
        stroke: '#333333',
        strokeWidth: 2,
        scaleY: 1
    });
    var dimension_group = new fabric.Group([dimension_mark, group_name], {
        left: 50,
        top: 50
    });
    canvas.add(dimension_group);
    dimension_group.on('mousedown', fabricDblClick(dimension_group, function (obj) {
        ungrouping(dimension_group);
        canvas.setActiveObject(group_name);
        group_name.enterEditing();
        group_name.selectAll();
    }));
}
addRuler();
