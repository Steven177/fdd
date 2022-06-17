(function () {
    var searchTerm, panelContainerId;
    // Create a new contains that is case insensitive
    $.expr[':'].containsCaseInsensitive = function (n, i, m) {
        return jQuery(n).text().toUpperCase().indexOf(m[3].toUpperCase()) >= 0;
    };
    $("#accordion_search_bar").on("change keyup paste click", function () {
        searchTerm = $(this).val();
      console.log(searchTerm);
        $("#accordionPanelsStayOpenExample > .accordion-item").each(function () {
            panelContainerId = "#" + $(this).attr("id");
          console.log(panelContainerId)

          $(panelContainerId + ":not(:containsCaseInsensitive(" + searchTerm + "))").hide();

            $(panelContainerId + ":containsCaseInsensitive(" + searchTerm + ")").show();
        });
    });
})();

// https://stackoverflow.com/questions/31799326/javascript-error-cannot-read-property-of-undefined
// https://codepen.io/Steven177/pen/VwQRGZq

