'use strict'
;(function() {
  let drawing = {
    backgroundImage: undefined,
    elements: []
  }

  const adjustAspectRatio = () => {
    // Adjust so we don't lose aspect ratio on resize
    const canvas = document.getElementById('result-canvas')
    const trueAR = canvas.width / canvas.height
    canvas.style.maxHeight = canvas.clientWidth / trueAR
  }

  const drawBoundingBoxes = function(ctx, probThresold) {
    const outlineWidth = 1
    const fontSize = 12
    const pad = 5

    ctx.lineWidth = outlineWidth

    // Draw actual bounding boxes
    drawing.elements.forEach(elem => {
      if (elem.prob < probThresold) {
        return
      }

      ctx.fillStyle = elem.fillColor
      ctx.fillRect(elem.x, elem.y, elem.width, elem.height)

      ctx.strokeStyle = elem.outlineColor
      ctx.strokeRect(elem.x, elem.y, elem.width, elem.height)
    })

    // Draw labels
    ctx.font = 'bold ' + fontSize + 'px Quicksand'
    ctx.textBaseline = 'bottom'
    drawing.elements.forEach(elem => {
      if (elem.prob < probThresold) {
        return
      }

      const text = elem.label + ' ' + elem.prob.toFixed(2)
      const textWidth = ctx.measureText(text).width

      ctx.fillStyle = elem.labelColor
      ctx.fillRect(
        elem.x,
        elem.y + elem.height - fontSize - pad * 2 - outlineWidth,
        textWidth + pad * 2 + outlineWidth,
        fontSize + pad * 2 + outlineWidth
      )

      ctx.fillStyle = 'rgba(30, 30, 30, 1.0)'
      ctx.fillText(text, elem.x + pad, elem.y + elem.height - pad)
    })
  }

  function draw(probThresold) {
    const canvas = document.getElementById('result-canvas')
    const ctx = canvas.getContext('2d')
    const image = drawing.backgroundImage

    // Don't upscale images
    canvas.style.maxWidth = image.width
    canvas.style.maxHeight = image.height

    ctx.canvas.width = image.width
    ctx.canvas.height = image.height
    ctx.drawImage(image, 0, 0, image.width, image.height)

    drawBoundingBoxes(ctx, probThresold)
    adjustAspectRatio()
  }

  function drawImage(imageFile) {
    const slider = document.getElementById('prob-threshold')
    const probThresold = slider.value / 100

    const image = new Image()
    image.src = URL.createObjectURL(imageFile)
    image.onload = function() {
      URL.revokeObjectURL(this.src) // release object to avoid memory leak

      drawing.backgroundImage = image
      draw(probThresold)

      window.addEventListener('resize', function() {
        draw(probThresold)
      })

      slider.addEventListener('input', function(event) {
        draw(event.target.value / 100)
      })
    }
  }

  const storeElementsToDraw = function(objects, probs, labels) {
    let i

    let labelHexColors = {}

    // Generate unique set of labels
    for (i = 0; i < labels.length; i++) {
      labelHexColors[labels[i]] = null
    }
    const distinctLabels = Object.keys(labelHexColors)

    // Generate color palette based on # of distinct labels,
    // using Google's palette generator script.
    const colors = window.palette('mpn65', distinctLabels.length)
    for (i = 0; i < distinctLabels.length; i++) {
      labelHexColors[distinctLabels[i]] = colors[i]
    }

    drawing.elements = []
    for (i = 0; i < objects.length; i++) {
      drawing.elements.push({
        prob: probs[i],
        label: labels[i],
        fillColor: window.hexToRgba(labelHexColors[labels[i]], 0.1),
        outlineColor: window.hexToRgba(labelHexColors[labels[i]], 1.0),
        labelColor: window.hexToRgba(labelHexColors[labels[i]], 0.5),
        x: objects[i][0],
        y: objects[i][1],
        width: objects[i][2] - objects[i][0],
        height: objects[i][3] - objects[i][1]
      })
    }
  }

  const formSubmit = function(form) {
    const loading = document.getElementById('loading')
    const responseDiv = document.getElementById('response')
    const canvas = document.getElementById('result-canvas')

    var formdata = new FormData(form)
    const url = '/api/fasterrcnn/predict'

    const xhr = new XMLHttpRequest()
    xhr.open('POST', url, true)
    xhr.send(formdata)

    responseDiv.style.display = 'none'
    loading.style.display = 'flex'
    canvas.style.display = 'none'

    xhr.onreadystatechange = function() {
      if (xhr.readyState == 4) {
        const response = JSON.parse(xhr.response)

        if (xhr.status == 200) {
          storeElementsToDraw(
            response.objects,
            response.objects_labels_prob,
            response.objects_labels
          )

          drawImage(formdata.getAll('image')[0])
          canvas.style.display = ''
          document.getElementById('result-separator').style.display = 'initial'
        }

        document.getElementById('api-response').innerHTML = xhr.response
        responseDiv.style.display = 'inline'
      }

      loading.style.display = 'none'
    }
  }

  window.addEventListener('resize', adjustAspectRatio)

  document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('image-form')
    form.addEventListener('submit', event => {
      event.preventDefault()
      formSubmit(form)
    })

    document
      .getElementById('prob-threshold')
      .addEventListener('input', function(event) {
        document.getElementById('prob-threshold-value').innerHTML = (
          event.target.value / 100
        ).toFixed(2)
      })
  })
})()
