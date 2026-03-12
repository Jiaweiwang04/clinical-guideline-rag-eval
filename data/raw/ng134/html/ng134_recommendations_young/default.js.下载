$(function() {

  var hText = "Hide ";
  var sText = "Show ";
  var $cl = $('.collapse-links');

  $cl.on('show', function() {
    var $ct = $('#chapter-toggle');
    var text = $ct.html();
    if (text.startsWith(sText) || text.startsWith(hText)) {
  	text = text.substring(5);
  	$ct.html('Hide ' + text);
    }
    $ct.children('i').first().removeClass('icon-nice-chevron-down');
    $ct.children('i').first().addClass('icon-nice-chevron-up');
  });

  $cl.on('hide', function() {
    var $ct = $('#chapter-toggle');
    var text = $ct.html();
    if (text.startsWith(sText) || text.startsWith(hText)) {
  	text = text.substring(5);
    	$ct.html('Show ' + text);
    }
    $ct.children('i').first().removeClass('icon-nice-chevron-up');
    $ct.children('i').first().addClass('icon-nice-chevron-down');
  });

});

