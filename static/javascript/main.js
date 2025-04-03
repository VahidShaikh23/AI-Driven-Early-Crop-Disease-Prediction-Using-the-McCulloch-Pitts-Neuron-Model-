$(document).ready(function () {
    // Hide sections initially
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Function to preview uploaded image
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    // Trigger preview when an image is selected
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Handle predict button click
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make AJAX request to Flask API
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Hide loader and show results
                $('.loader').hide();
                $('#result').fadeIn(600);

                // Display prediction results
                $('#result').html(
                    '<b>Prediction:</b> ' + data.prediction + '<br>' +
                    '<b>Confidence:</b> ' + data.confidence + '<br>' +
                    '<b>Treatment:</b> ' + data.treatment
                );

                console.log('Prediction result:', data);
            },
            error: function (xhr, status, error) {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').html('<b>Error:</b> ' + xhr.responseText);
                console.error('Prediction error:', error);
            }
        });
    });
});


