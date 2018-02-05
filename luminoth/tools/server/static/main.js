// TODO: for now, don't care about > 9 classes per image (very rare).
const FILL_COLORS = [
  'rgba(230, 25, 75, 0.2)',
  'rgba(60, 180, 75, 0.2)',
  'rgba(0, 130, 200, 0.2)',
  'rgba(245, 130, 48, 0.2)',
  'rgba(145, 30, 180, 0.2)',
  'rgba(210, 245, 60, 0.2)',
  'rgba(230, 190, 255, 0.2)',
  'rgba(128, 0, 0, 0.2)',
  'rgba(170, 255, 195, 0.2)'
]

const Shape = function(x, y, w, h, fill) {
  this.x = x || 0
  this.y = y || 0
  this.w = w || 0
  this.h = h || 0

  this.fill = fill || 'rgba(255, 255, 255, 0)'
}

Shape.prototype.draw = function(ctx) {
  ctx.fillStyle = this.fill
  ctx.fillRect(this.x, this.y, this.w, this.h)

  ctx.strokeStyle = '#F00'
  ctx.lineWidth = 1
  ctx.strokeRect(this.x, this.y, this.w, this.h)
}

function rand(min, max) {
  return Math.floor(Math.random() * (max - min + 1) + min)
}

function drawCanvas(canvas, imageFile, objects, probs, labels, minProb) {
  ctx = canvas.getContext('2d')

  image = new Image()
  image.src = URL.createObjectURL(imageFile)
  image.onload = function() {
    URL.revokeObjectURL(this.src) // release object to avoid memory leak

    var imageWidth = image.width
    var imageHeight = image.height

    ctx.canvas.width = imageWidth
    ctx.canvas.height = imageHeight

    ctx.drawImage(image, 0, 0, image.width, image.height)

    var labelColors = {}
    for (var i = 0; i < labels.length; i++) {
      var label = labels[i]
      if (!(label in labelColors)) {
        labelColors[label] =
          FILL_COLORS[
            Object.keys(labelColors).length % Object.keys(FILL_COLORS).length
          ]
      }
    }

    for (var i = 0; i < objects.length; i++) {
      var prob = probs[i]
      if (prob < minProb) {
        continue
      }
      var label = labels[i]
      var x = objects[i][0]
      var y = objects[i][1]
      var width = objects[i][2] - objects[i][0]
      var height = objects[i][3] - objects[i][1]
      ctx.font = '10px serif'
      ctx.fillStyle = 'rgba(30, 30, 30, 1.0)'
      ctx.fillText(label, x, y - 3)
      ctx.fillStyle = labelColors[label]
      ctx.fillRect(x, y, width, height)
      ctx.strokeStyle = 'rgba(230, 20, 20, 0.55)'
      ctx.lineWidth = 1
      ctx.strokeRect(x, y, width, height)
    }
  }
}

var formSubmit = function(form) {
  var modelType = document.getElementById('modelField').value
  var total = document.getElementById('totalField').value
  var minProb = document.getElementById('minField').value
  var resultsDiv = document.getElementById('results-row')
  var jsonDiv = document.getElementById('results')
  var imageCanvas = document.getElementById('image-canvas')

  var xhr = new XMLHttpRequest()
  var formdata = new FormData(form)
  url = '/api/' + modelType + '/predict'
  if (total) {
    url += '?total=' + total
  }

  xhr.open('POST', url, true)
  xhr.send(formdata)

  xhr.onreadystatechange = function() {
    if (xhr.readyState == 4 && xhr.status == 200) {
      jsonDiv.innerHTML = xhr.response
      response = JSON.parse(xhr.response)
      resultsDiv.style.display = ''

      drawCanvas(
        imageCanvas,
        formdata.getAll('image')[0],
        response.objects,
        response.objects_labels_prob,
        response.objects_labels,
        minProb
      )
    }
  }
}

document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('imageForm')
  form.addEventListener('submit', function(event) {
    event.preventDefault()
    formSubmit(form)
  })
})
