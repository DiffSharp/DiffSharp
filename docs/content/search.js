var lunrIndex, pagesIndex;

function endsWith(str, suffix) {
    return str.indexOf(suffix, str.length - suffix.length) !== -1;
}

// Initialize lunrjs using our generated index file
function initLunr() {
    if (!endsWith(baseurl,"/")){
        baseurl = baseurl+'/'
    };

    // First retrieve the index file
    $.getJSON(baseurl +"index.json")
        .done(function(index) {
            pagesIndex = index;
            // Set up lunrjs by declaring the fields we use
            // Also provide their boost level for the ranking
            lunrIndex = lunr(function() {
                this.ref("uri");
                this.field('title', {
		    boost: 15
                });
                this.field('tags', {
		    boost: 10
                });
                this.field("content", {
		    boost: 5
                });
				
                this.pipeline.remove(lunr.stemmer);
                this.searchPipeline.remove(lunr.stemmer);
				
                // Feed lunr with each file and let lunr actually index them
                pagesIndex.forEach(function(page) {
		    this.add(page);
                }, this);
            })
        })
        .fail(function(jqxhr, textStatus, error) {
            var err = textStatus + ", " + error;
            console.error("Error getting Hugo index file:", err);
        });
}
console.warn("Loading search.js");

/**
 * Trigger a search in lunr and transform the result
 *
 * @param  {String} query
 * @return {Array}  results
 */
function search(queryTerm) {
    // Find the item in our index corresponding to the lunr one to have more info
    console.warn("search.js: searching for: ", queryTerm);
    return lunrIndex.search(queryTerm+"^100"+" "+queryTerm+"*^10"+" "+"*"+queryTerm+"^10"+" "+queryTerm+"~2^1").map(function(result) {
            return pagesIndex.filter(function(page) {
                return page.uri === result.ref;
            })[0];
        });
}
console.warn("search.js: initLunr");

// Let's get started
initLunr();

console.warn("search.js: after initLunr");
$( document ).ready(function() {
    console.warn("search.js: document ready");
    var searchList = new autoComplete({
        /* selector for the search box element */
        minChars: 1,
        selector: $("#search-by").get(0),
        /* source is the callback to perform the search */
        source: function(term, response) {
            console.warn("search.js: source callback, term = ", term);
            response(search(term));
        },
        /* renderItem displays individual search results */
        renderItem: function(item, search) {
            search = search.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
            var re = new RegExp("(" + search.split(' ').join('|') + ")", "gi");
            return '<div class="autocomplete-suggestion" data-val="'+search+'" data-uri="' + item.uri + '">' + item.title.replace(re, "<b>$1</b>") + '</div>';
        },
        /* onSelect callback fires when a search suggestion is chosen */
        onSelect: function(e, term, item) {
            console.warn("search.js: onSelect, location.href = ", item.getAttribute('data-uri'));
            location.href = item.getAttribute('data-uri');
        }
    });
    // var ajax;
    // jQuery('[data-search-input]').on('input', function() {
    //     var input = jQuery(this),
    //         value = input.val(),
    //         items = jQuery('[data-nav-id]');
    //     items.removeClass('search-match');
    //     if (!value.length) {
    //         $('ul.topics').removeClass('searched');
    //         items.css('display', 'block');
    //         sessionStorage.removeItem('search-value');
    //         $(".highlightable").unhighlight({ element: 'mark' })
    //         return;
    //     }

    //     sessionStorage.setItem('search-value', value);
    //     $(".highlightable").unhighlight({ element: 'mark' }).highlight(value, { element: 'mark' });

    //     if (ajax && ajax.abort) ajax.abort();

    //     jQuery('[data-search-clear]').on('click', function() {
    //         jQuery('[data-search-input]').val('').trigger('input');
    //         sessionStorage.removeItem('search-input');
    //         $(".highlightable").unhighlight({ element: 'mark' })
    //     });
    // });
});
