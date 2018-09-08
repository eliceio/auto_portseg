$(document).ready(function() {
    function preview_img(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('.img-show').html("<img id='user_img' src='" + e.target.result + "' alt='user_img' name='user_img'>");
                $('.img-show').css('border', 'none');
                $('.progress-p').css('visibility', 'visible');
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $('#id_image_file').on('change', function () {
        preview_img(this);
        var file = this.files[0];
        var form_data = new FormData();
        form_data.append('file', file);
    });
});