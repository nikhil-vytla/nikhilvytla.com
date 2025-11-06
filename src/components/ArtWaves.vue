<script setup lang='ts'>
const el = ref<HTMLCanvasElement | null>(null)
const { random, sin, PI } = Math
const size = reactive(useWindowSize())

// Configuration
const WAVE_COUNT = 6
const WAVE_SPEED = 0.002
const WAVE_HEIGHT = 40
const WAVE_FREQUENCY = 0.008
const COLOR_ALPHA = 0.15

function initCanvas(canvas: HTMLCanvasElement, width = 400, height = 400, _dpi?: number) {
  const ctx = canvas.getContext('2d')!

  const dpr = window.devicePixelRatio || 1
  // @ts-expect-error vendor
  const bsr = ctx.webkitBackingStorePixelRatio || ctx.mozBackingStorePixelRatio || ctx.msBackingStorePixelRatio || ctx.oBackingStorePixelRatio || ctx.backingStorePixelRatio || 1

  const dpi = _dpi || dpr / bsr

  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`
  canvas.width = dpi * width
  canvas.height = dpi * height
  ctx.scale(dpi, dpi)

  return { ctx, dpi }
}

onMounted(async () => {
  const canvas = el.value!
  const { ctx } = initCanvas(canvas, size.width, size.height)

  let animationId: number
  let time = 0

  // Create wave parameters
  const waves = Array.from({ length: WAVE_COUNT }, (_, i) => ({
    amplitude: WAVE_HEIGHT * (0.6 + random() * 0.4),
    frequency: WAVE_FREQUENCY * (0.8 + random() * 0.4),
    phase: random() * PI * 2,
    speed: WAVE_SPEED * (0.7 + random() * 0.6),
    yOffset: (size.height / (WAVE_COUNT + 1)) * (i + 1),
    hue: (200 + (i * 40)) % 360, // Blue to purple to pink spectrum
    opacity: 0.6 + random() * 0.4,
  }))

  const animate = () => {
    // Clear completely for crisp waves without grey buildup
    ctx.clearRect(0, 0, size.width, size.height)

    time += 1

    waves.forEach((wave, index) => {
      // Main wave line
      ctx.beginPath()
      ctx.strokeStyle = `hsla(${wave.hue}, 60%, 50%, ${COLOR_ALPHA * wave.opacity})`
      ctx.lineWidth = 1.5

      let firstPoint = true

      // Draw the wave with multiple harmonics for complexity
      for (let x = 0; x <= size.width; x += 3) {
        const baseY = wave.yOffset
          + wave.amplitude * sin(x * wave.frequency + wave.phase + time * wave.speed)

        // Add harmonics for more interesting wave shapes
        const harmonic1 = (wave.amplitude * 0.3) * sin(x * wave.frequency * 2.1 + time * wave.speed * 1.3)
        const harmonic2 = (wave.amplitude * 0.15) * sin(x * wave.frequency * 3.7 + time * wave.speed * 0.7)

        const y = baseY + harmonic1 + harmonic2

        if (firstPoint) {
          ctx.moveTo(x, y)
          firstPoint = false
        }
        else {
          ctx.lineTo(x, y)
        }
      }

      ctx.stroke()

      // Add flowing particles along the wave
      if (time % 15 === 0) {
        for (let i = 0; i < 2; i++) {
          const x = random() * size.width
          const baseY = wave.yOffset
            + wave.amplitude * sin(x * wave.frequency + wave.phase + time * wave.speed)
          const harmonic1 = (wave.amplitude * 0.3) * sin(x * wave.frequency * 2.1 + time * wave.speed * 1.3)
          const y = baseY + harmonic1

          ctx.beginPath()
          ctx.fillStyle = `hsla(${wave.hue}, 70%, 60%, ${COLOR_ALPHA * 1.5})`
          ctx.arc(x, y, 1.5, 0, PI * 2)
          ctx.fill()
        }
      }

      // Add subtle glow effect
      if (index % 2 === 0) {
        ctx.beginPath()
        ctx.strokeStyle = `hsla(${wave.hue}, 80%, 70%, ${COLOR_ALPHA * 0.3})`
        ctx.lineWidth = 3

        firstPoint = true
        for (let x = 0; x <= size.width; x += 4) {
          const y = wave.yOffset
            + wave.amplitude * sin(x * wave.frequency + wave.phase + time * wave.speed)
            + (wave.amplitude * 0.3) * sin(x * wave.frequency * 2.1 + time * wave.speed * 1.3)

          if (firstPoint) {
            ctx.moveTo(x, y)
            firstPoint = false
          }
          else {
            ctx.lineTo(x, y)
          }
        }

        ctx.stroke()
      }
    })

    animationId = requestAnimationFrame(animate)
  }

  // Start animation
  animate()

  // Handle window resize
  const handleResize = () => {
    initCanvas(canvas, size.width, size.height)
    // Update wave y offsets for new height
    waves.forEach((wave, i) => {
      wave.yOffset = (size.height / (WAVE_COUNT + 1)) * (i + 1)
    })
  }

  useEventListener('resize', handleResize)

  // Cleanup
  onUnmounted(() => {
    if (animationId) {
      cancelAnimationFrame(animationId)
    }
  })
})
</script>

<template>
  <div
    class="fixed top-0 bottom-0 left-0 right-0 pointer-events-none print:hidden dark:invert"
    style="z-index: -1"
  >
    <canvas ref="el" width="400" height="400" />
  </div>
</template>
