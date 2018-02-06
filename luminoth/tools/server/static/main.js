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

const drawBoundingBoxes = function(ctx, objects, probs, labels, minProb) {
  let labelColors = {}
  let i

  for (i = 0; i < labels.length; i++) {
    let label = labels[i]
    if (!(label in labelColors)) {
      labelColors[label] =
        FILL_COLORS[
          Object.keys(labelColors).length % Object.keys(FILL_COLORS).length
        ]
    }
  }

  for (i = 0; i < objects.length; i++) {
    let prob = probs[i]
    if (prob < minProb) {
      continue
    }

    let label = labels[i]
    let x = objects[i][0]
    let y = objects[i][1]
    let width = objects[i][2] - objects[i][0]
    let height = objects[i][3] - objects[i][1]

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

function drawImage(imageFile, objects, probs, labels, minProb) {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  const image = new Image()
  image.src = URL.createObjectURL(imageFile)
  image.onload = function() {
    URL.revokeObjectURL(this.src) // release object to avoid memory leak

    ctx.canvas.width = image.width
    ctx.canvas.height = image.height
    ctx.drawImage(image, 0, 0, image.width, image.height)

    drawBoundingBoxes(ctx, objects, probs, labels, minProb)

    // Get result as PNG and assign to element in DOM
    const domImg = document.getElementById('result-image')
    domImg.src = canvas.toDataURL()
  }
}

const formSubmit = function(form) {
  var modelType = document.getElementById('modelField').value
  var total = document.getElementById('totalField').value
  var minProb = document.getElementById('minField').value
  // var resultsDiv = document.getElementById('results-row')
  // var jsonDiv = document.getElementById('results')

  var formdata = new FormData(form)
  url = '/api/' + modelType + '/predict'
  if (total) {
    url += '?total=' + total
  }

  const xhr = new XMLHttpRequest()
  xhr.open('POST', url, true)
  xhr.send(formdata)

  xhr.onreadystatechange = function() {
    if (xhr.readyState == 4 && xhr.status == 200) {
      // jsonDiv.innerHTML = xhr.response
      const response = JSON.parse(xhr.response)
      // resultsDiv.style.display = ''

      drawImage(
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
