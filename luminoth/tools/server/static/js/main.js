'use strict'
;(function() {
  const drawBoundingBoxes = function(ctx, objects, probs, labels, minProb) {
    let labelColors = {}
    let i

    // Generate unique set of labels
    for (i = 0; i < labels.length; i++) {
      labelColors[labels[i]] = null
    }
    const distinctLabels = Object.keys(labelColors)

    // Generate color palette based on # of distinct labels
    const colors = palette('mpn65', distinctLabels.length).map(hex =>
      hexToRgba(hex, 0.25)
    )
    for (i = 0; i < distinctLabels.length; i++) {
      labelColors[distinctLabels[i]] = colors[i]
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
    const canvas = document.getElementById('result-canvas')
    const ctx = canvas.getContext('2d')

    const image = new Image()
    image.src = URL.createObjectURL(imageFile)
    image.onload = function() {
      URL.revokeObjectURL(this.src) // release object to avoid memory leak

      // Don't upscale images
      canvas.style.maxWidth = image.width
      canvas.style.maxHeight = image.height

      ctx.canvas.width = image.width
      ctx.canvas.height = image.height
      ctx.drawImage(image, 0, 0, image.width, image.height)

      drawBoundingBoxes(ctx, objects, probs, labels, minProb)
    }
  }

  const formSubmit = function(form) {
    let modelType = document.getElementById('modelField').value
    let total = document.getElementById('totalField').value
    let minProb = document.getElementById('minField').value
    // let resultsDiv = document.getElementById('results-row')
    // let jsonDiv = document.getElementById('results')

    var formdata = new FormData(form)
    let url = '/api/' + modelType + '/predict'
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
    form.addEventListener('submit', event => {
      event.preventDefault()
      formSubmit(form)
    })
  })
})()
