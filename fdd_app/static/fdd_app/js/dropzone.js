Dropzone.autoDiscover=false;
const myDropzone= new Dropzone('#my-dropzone',{
    url:'images/',
    maxFiles:10,
    maxFilesize:2,
    acceptedFiles:'.jpg', '.jpeg', '.png',
})
