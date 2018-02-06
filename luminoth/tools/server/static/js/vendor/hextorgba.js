//
// From: https://raw.githubusercontent.com/misund/hex-to-rgba
//
;(function() {
  const removeHash = hex => (hex.charAt(0) === '#' ? hex.slice(1) : hex)

  const parseHex = nakedHex => {
    const isShort = 3 === nakedHex.length || 4 === nakedHex.length

    const twoDigitHexR = isShort
      ? `${nakedHex.slice(0, 1)}${nakedHex.slice(0, 1)}`
      : nakedHex.slice(0, 2)
    const twoDigitHexG = isShort
      ? `${nakedHex.slice(1, 2)}${nakedHex.slice(1, 2)}`
      : nakedHex.slice(2, 4)
    const twoDigitHexB = isShort
      ? `${nakedHex.slice(2, 3)}${nakedHex.slice(2, 3)}`
      : nakedHex.slice(4, 6)
    const twoDigitHexA =
      (isShort
        ? `${nakedHex.slice(3, 4)}${nakedHex.slice(3, 4)}`
        : nakedHex.slice(6, 8)) || 'ff'

    // const numericA = +((parseInt(a, 16) / 255).toFixed(2));

    return {
      r: twoDigitHexR,
      g: twoDigitHexG,
      b: twoDigitHexB,
      a: twoDigitHexA
    }
  }

  const hexToDecimal = hex => parseInt(hex, 16)

  const hexesToDecimals = ({ r, g, b, a }) => ({
    r: hexToDecimal(r),
    g: hexToDecimal(g),
    b: hexToDecimal(b),
    a: +(hexToDecimal(a) / 255).toFixed(2)
  })

  const isNumeric = n => !isNaN(parseFloat(n)) && isFinite(n)

  const formatRgb = (decimalObject, parameterA) => {
    const { r, g, b, a: parsedA } = decimalObject
    const a = isNumeric(parameterA) ? parameterA : parsedA

    return `rgba(${r}, ${g}, ${b}, ${a})`
  }

  /**
   * Turns an old-fashioned css hex color value into a rgb color value.
   *
   * If you specify an alpha value, you'll get a rgba() value instead.
   *
   * @param The hex value to convert. ('123456'. '#123456', ''123', '#123')
   * @param An alpha value to apply. (optional) ('0.5', '0.25')
   * @return An rgb or rgba value. ('rgb(11, 22, 33)'. 'rgba(11, 22, 33, 0.5)')
   */
  const hexToRgba = (hex, a) => {
    const hashlessHex = removeHash(hex)
    const hexObject = parseHex(hashlessHex)
    const decimalObject = hexesToDecimals(hexObject)

    return formatRgb(decimalObject, a)
  }
  window.hexToRgba = hexToRgba
})()
