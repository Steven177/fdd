const jq = document.createElement('script');
jq.src = "https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js";
document.head.appendChild(jq);
jq.addEventListener('load', () => {

});

$(document).ready(function() {
    var searchTerm, panelContainerId;
    // Create a new contains that is case insensitive
    $.expr.pseudos.containsCaseInsensitive = function (n, i, m) {
        return jQuery(n).text().toUpperCase().indexOf(m[3].toUpperCase()) >= 0;
    };
    $("#accordion_search_bar").on("change keyup paste click", function () {
        searchTerm = $(this).val();
        $("#accordionPanelsStayOpenExample > .accordion-item").each(function () {
            panelContainerId = "#" + $(this).attr("id");

          $(panelContainerId + ":not(:containsCaseInsensitive(" + searchTerm + "))").hide();

            $(panelContainerId + ":containsCaseInsensitive(" + searchTerm + ")").show();
        });
    });
});


