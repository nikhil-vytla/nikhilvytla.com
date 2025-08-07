---
title: Supporting OSS
date: 2023-05-31
lang: en
duration: 12min
description: Open-source projects form the backbone of the internet, and it feels more important than ever before to support them.
---

[[toc]]

Every day, billions of people rely on software that runs the modern world. From the web browsers displaying this text to the servers hosting it, from the networks routing data packets to the cryptographic libraries securing transactions, nearly every digital interaction depends on open-source software (OSS).

Supporting open-source projects can happen in a few ways. Contributing code, fixing bugs, and discussing feature requests are great ways for folks to get involved, especially budding new software engineers.

I'd like to also bring attention to a less seen but equally as important facet of support: **funding**. I've had the pleasure of contributing to and maintaining open-source libraries over the years, and while I can't speak for everyone, I believe that most open-source maintainers do it for the [mission](https://opensource.org/about#:~:text=source%20ecosystem%20thrives.-,Mission,-The%20Open%20Source), not for the money.

That being said, most projects do operate on financial life support, maintained by volunteers or small teams that still need to keep the lights on and make a living so that they can (hopefully) spend more time maintaining and building their projects. Money doesn't grow on trees, and it sure doesn't grow in GitHub issue comments.

## Underappreciated Foundations

I'm not exactly sure why, but open-source software seems to operates on a paradox: the more critical a project becomes, the more invisible it often is to end users.

Consider these statistics:

- **SQLite** is embedded in every iPhone, Android device, and web browser, yet runs on donations
- **OpenSSL** secures nearly every HTTPS connection on the internet with a team smaller than most coffee shops
- **curl** powers data transfer for billions of applications but relies entirely on volunteer contributions
- **FFmpeg** processes virtually every video you watch online, and is maintained by a handful of developers

There's a famous [XKCD comic](https://www.explainxkcd.com/wiki/index.php/2347:_Dependency) that illustrates this:

![XKCD #2347: Dependency](https://www.explainxkcd.com/wiki/images/d/d7/dependency.png 'XKCD #2347: Dependency')

A brief aside on types of open-source projects:

Some projects are more "user-facing". These projects are directly interacted with by developers and users daily, and benefit from large community forums and active discussions.

Others are "below-the-surface". These are usually projects that most devs indirectly depend on, but don't realize!

In a similar vein to project/code visibility, some project maintainers and communities are more vocal about needing funding/financial support, and others (for a variety of reasons) are perhaps less so.

Regardless of their levels of visibility, at the end of the day, both types of projects have three things in common:

1. they are vital to the OSS ecosystem,
2. they rely on small, dedicated teams of volunteers to continue building safer, scalable, more robust, and frankly kick-ass software that benefits all of us, and
3. they deserve our support. 💪

## Jenga Blocks All the Way Down

Companies (many of them [household names](https://netflixtechblog.com/why-we-use-and-contribute-to-open-source-software-1faa77c2e5c4)) generate trillions in revenue using free open-source software, yet most contribute a fraction back to the projects they depend on. In today's chronically online world, this funding imbalance impacts both our technical capabilities and our economic strength.

In short:

1. Without sustainable funding, critical projects depend on the goodwill of overworked volunteers who often quit, increasing attack vector space via security holes and technical debt.

2. Underfunded projects lack resources for proper security audits, testing infrastructure, and timely vulnerability responses.

3. Without funding for new features and optimizations, critical infrastructure stagnates, limiting technological innovation across the entire ecosystem.

This results in:

- **Large positive externalities**: One open-source project can create billions in economic value
- **Tragedy of the commons**: Everyone benefits from open-source but few contribute to maintenance
- **Single points of failure**: Key maintainers become irreplaceable bottlenecks

Some innovative organizations have begun addressing this. [ESLint forwards sponsorships to dependencies](https://eslint.org/blog/2022/02/paying-contributors-sponsoring-projects/), and the [Open Source Security Foundation](https://openssf.org/) coordinates security improvements. These efforts are promising, but they're only the start!

## A List of Projects to Consider Supporting

The following list contains examples of projects that form different parts of the backbone of our digital infrastructure and the open internet. Note that this is a non-exhaustive list.

### Core Infrastructure & Cryptography

**[OpenSSL](https://www.openssl.org/)**

- **What it does**: Provides cryptographic functionality for virtually every secure web connection
- **Impact**: Secures trillions of dollars in online transactions daily
- **Funding**: [GitHub Sponsors](https://github.com/sponsors/openssl) | [OpenSSL Foundation](https://openssl-foundation.org/donate/)

**[curl](https://curl.se/)**

- **What it does**: Command-line tool and library for transferring data with URLs
- **Impact**: Used by virtually every programming language and application for HTTP requests
- **Funding**: [GitHub Sponsors](https://github.com/sponsors/bagder) | [Open Collective](https://opencollective.com/curl)

### Internet Infrastructure

**[Let's Encrypt](https://letsencrypt.org/)**

- **What it does**: Free, automated certificate authority providing TLS certificates
- **Impact**: Enabled HTTPS for 95% of all websites, securing billions of connections
- **Funding**: [Donations](https://letsencrypt.org/donate/) | [Corporate sponsorship](https://letsencrypt.org/become-a-sponsor/)

**[Wikimedia Foundation](https://wikimediafoundation.org/)**

- **What it does**: Operates Wikipedia and related knowledge projects
- **Impact**: Provides free access to human knowledge for billions of people
- **Funding**: [Individual donations](https://donate.wikimedia.org/) | Corporate partnerships

**[Internet Archive](https://archive.org/)**

- **What it does**: Digital library preserving websites, books, movies, music, and software
- **Impact**: Preserves digital heritage and provides free access to historical information
- **Funding**: [Donations](https://archive.org/donate/) | Grants

### Development Tools & Libraries

**[FFmpeg](https://ffmpeg.org/)**

- **What it does**: Multimedia framework for recording, converting, and streaming audio/video
- **Impact**: Powers virtually every video platform, from YouTube to Netflix
- **Funding**: [Donations](https://ffmpeg.org/donations.html)

### Programming Language Foundations

**[Python Software Foundation](https://www.python.org/psf/)**

- **What it does**: Supports development of the Python programming language
- **Impact**: Python powers AI/ML, web development, and scientific computing globally
- **Funding**: [Donations](https://www.python.org/psf/donations/) | [Corporate sponsorship](https://www.python.org/psf/sponsorship/)

## The Path Forward

Broadly, I believe it's important to recognize that open-source infrastructure is a public good that deserves to be invested in, just like roads, bridges, and utilities.

It's also equally important to show appreciation for OSS, to contribute and maintain code, and, if you are able to donate, please do!

For the record, I'm not proposing a silver bullet, and I also don't believe that it's only on the general population to "foot the bill". Large companies, especially those who benefit the most from OSS, need to step up to the plate.

Just like how [sidewalk curbs, initially designed to benefit vulnerable groups, have ended up benefitting all of society](https://ssir.org/articles/entry/the_curb_cut_effect), the impacts of open-source projects benefit diverse, interdisciplinary communities around the world! ◡̈

---

_The projects listed above represent just the tip of the iceberg. If you have another project you'd like for me to feature on this list, please let me know!_
