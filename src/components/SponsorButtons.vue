<template>
  <p>
    If you enjoy my work and find it useful, consider sponsoring me (or better yet, donate to a cause that you care about!)
  </p>

  <p flex="~ gap-2 wrap items-center">
    <a
      href="https://github.com/sponsors/nikhil-vytla"
      target="_blank"
      class="group btn-rose inline-block"
    >
      <div
        i-ph-heart-duotone
        group-hover="i-ph-heart-fill text-rose"
        transition-all duration-200 ease-out
      />
      Sponsor Nikhil on GitHub
    </a>
    <a
      href="https://donate.wikimedia.org/w/index.php"
      target="_blank"
      class="group btn-yellow inline-block"
    >
      <div
        i-ph-lightning-duotone
        group-hover="i-ph-lightning-fill text-yellow"
        transition-all duration-200 ease-out
      />
      Donate to Wikimedia
    </a>
    <!-- TODO: add a link to a post on organizations I like donating to -->
    <span op50 text-sm>
      <a href="/posts/sponsorship-forwarding" target="_blank">How does this work?</a>
    </span>
  </p>
</template>
