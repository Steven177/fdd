
$('.failure_severity_form').submit(function(event){
    event.preventDefault();
    var  severity = $(".failure_severity", this).val();
});

console.log(severity)
