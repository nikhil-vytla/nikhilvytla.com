---
title: Projects - Nikhil Vytla
display: Projects
description: A list of projects that I am proud of
wrapperClass: 'text-center'
art: dots
# TODO: add projects
# TODO: emojis should be sourced from https://icones.js.org/
# NOTE: the syntax is `i-<provider>-<emojiname>` (e.g. for provider: "emojione" and emojiname: "squid" --> `i-emojione-squid`)
# NOTE: `saturate-0` grayscales the emoji
#
projects:
  Current Focus:
    - name: 'TruLens'
      link: 'https://github.com/truera/trulens'
      desc: 'Evaluate and track LLMs/Agents (powered by TruEra and Snowflake)'
      icon: 'i-emojione-squid'
      # icon: 'i-simple-icons-snowflake'
    - name: 'Environmental Sustainability'
      link: 'https://github.com/cncf/tag-env-sustainability'
      desc: 'Cloud Native Compute Foundation (CNCF) Technical Advisory Group (TAG)'
      icon: 'i-simple-icons-cncf'

  Harvard:
    - name: 'Intro to Data Science'
      link: 'https://github.com/datasciencelabs/2024'
      desc: 'BST 260, Fall 2024'
      icon: 'i-fluent-hat-graduation-12-regular'

  UNC:
    - name: 'Resume Database'
      link: 'https://github.com/nikhil-vytla/unc-cs-resume-db'
      desc: 'Enabling UNC CS Students to find internship and job opportunities'
      icon: 'i-fluent-emoji-flat-ram'
    - name: 'Geospatial Store'
      link: 'https://github.com/akan72/geospatial-store'
      desc: 'Making Earth observation more accessible for everyone'
      icon: 'i-mdi-earth-arrow-up'
---

<!-- @layout-full-width -->
<ListProjects :projects="frontmatter.projects" />
