---
title: Site Maintenance & TODOs
date: 2025-07-04T16:00:00Z
lang: en
duration: 10min
type: note
---

## TODOs

- Migrate blog posts from [interactive-blog](https://nikhil-vytla.github.io/interactive-blog) to [/posts](/posts).

  - Enable pretty printing + line highlighting with Shiki
  - Replace custom Pyodide implementation with Marimo notebooks (either widgets/iframes or [islands 🏝️](https://docs.marimo.io/guides/island_example/) (experimental feat))

- Upload updated resume
- Add more photos
- Update projects (and project categories to reflect topics) and demos page with additional repos/videos
- Update bookmarks
- Update media
- Confirm that math works (may need to fiddle with mathjax3 CSS more!)
- Confirm that code renders work

### Testing code renders

> [!NOTE]
> Only some transformers work, it seems like only [!code word:Hello] (word highlight) and [!code hl] (line highlight) for the time being

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit([[1], [2], [3]], [2, 4, 6]) # [!code hl]
print(model.predict([[4]]))  # Output: [8]
```

```python
# [!code word:RandomForestClassifier]
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
print(RandomForestClassifier().fit(X, y).score(X, y))  # ~0.97
```

```ts
console.log('No errors or warnings')
console.error('Error') // [!code error]
console.warn('Warning') // [!code warning]
```

> [!NOTE]
> Twoslash only work for code written in ts

```ts twoslash
interface Todo {
  title: string
}

const todo: Readonly<Todo> = {
  title: 'Delete inactive users',
//  ^?
}
```

### Testing math renders

if $t = 1$ and $f = 2$, and if $w = t + f$, then $w = 3$. Also:

When $a \ne 0$, there are two solutions to $(ax^2 + bx + c = 0)$ and they are:
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

#### Maxwell's Equations

| equation                                                                                                                                                                  | description                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| $\nabla \cdot \vec{\mathbf{B}}  = 0$                                                                                                                                      | divergence of $\vec{\mathbf{B}}$ is zero                                               |
| $\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t}  = \vec{\mathbf{0}}$                                                          | curl of $\vec{\mathbf{E}}$ is proportional to the rate of change of $\vec{\mathbf{B}}$ |
| $\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} = \frac{4\pi}{c}\vec{\mathbf{j}}    \nabla \cdot \vec{\mathbf{E}} = 4 \pi \rho$ | _???_                                                                                  |

![electricity](https://i.giphy.com/Gty2oDYQ1fih2.gif)

#### Einstein's Equations

$$
E = mc^2 \tag{1}
$$

$$
F = ma \tag{2}
$$

As we can see, force equals mass times acceleration.

#### Merge Sort

The time complexity of merge sort is $O(n \log n)$.

The recursive relation is:

$$
T(n) = \begin{cases}
1 & \text{if } n = 1 \\
2T(n/2) + n & \text{if } n > 1
\end{cases}
$$

Big-O notation: $f(n) = O(g(n))$ means $\exists c, n_0$ such that $f(n) \leq c \cdot g(n)$ for all $n \geq n_0$.

## Completed

- Move this code to new repo: [nikhil-vytla/nikhilvytla.com](https://github.com/nikhil-vytla/nikhilvytla.com)
- Setup Netlify
- Remove domain name redirect from old site: [nikhil-vytla.github.io](https://nikhil-vytla.github.io)
