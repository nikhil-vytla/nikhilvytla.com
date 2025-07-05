<template>
  <img src="/favicon.svg" alt="Logo - Nikhil Vytla" class="logo">
</template>

<style scoped>
@media (prefers-reduced-motion) {
  .logo {
    animation: none !important;
  }
}

@media print {
  .logo {
    animation: none !important;
  }
}

@keyframes pulse {
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.05);
    opacity: 0.9;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
}

.logo {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  animation: pulse 10s ease forwards infinite;
  transform-origin: center;
  margin-top: -15px;
  object-fit: cover;
}

.dark .logo {
  filter: invert(0);
}
</style>