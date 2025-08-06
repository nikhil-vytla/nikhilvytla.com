---
title: Markdown Syntax Guide
date: '2024-10-04'
lang: en
duration: 5min
type: note
---

[[toc]]

Here's some example Markdown syntax.

## Headings

The following HTML `<h1>`—`<h6>` elements represent six levels of section headings. `<h1>` is the highest section level while `<h6>` is the lowest.

<h1>H1</h1>

<h2>H2</h2>

<h3>H3</h3>

<h4>H4</h4>

<h5>H5</h5>

<h6>H6</h6>

<h7>H7</h7>

## Paragraph

Xerum, quo qui aut unt expliquam qui dolut labo. Aque venitatiusda cum, voluptionse latur sitiae dolessi aut parist aut dollo enim qui voluptate ma dolestendit peritin re plis aut quas inctum laceat est volestemque commosa as cus endigna tectur, offic to cor sequas etum rerum idem sintibus eiur? Quianimin porecus evelectur, cum que nis nust voloribus ratem aut omnimi, sitatur? Quiatem. Nam, omnis sum am facea corem alique molestrunt et eos evelece arcillit ut aut eos eos nus, sin conecerem erum fuga. Ri oditatquam, ad quibus unda veliamenimin cusam et facea ipsamus es exerum sitate dolores editium rerore eost, temped molorro ratiae volorro te reribus dolorer sperchicium faceata tiustia prat.

Itatur? Quiatae cullecum rem ent aut odis in re eossequodi nonsequ idebis ne sapicia is sinveli squiatum, core et que aut hariosam ex eat.

## Images

### Syntax

```markdown
![Alt text](./full/or/relative/path/of/image)
```

### Output

![electricity](https://i.giphy.com/Gty2oDYQ1fih2.gif)

## Blockquotes

The blockquote element represents content that is quoted from another source, optionally with citations which may be within a `footer` or `cite` element, and optionally with in-line changes such as annotations and abbreviations.

### Blockquote without attribution

#### Syntax

```markdown
> Tiam, ad mint andaepu dandae nostion secatur sequo quae.
> **Note** that you can use _Markdown syntax_ within a blockquote.
```

#### Output

> Tiam, ad mint andaepu dandae nostion secatur sequo quae.
> **Note** that you can use _Markdown syntax_ within a blockquote.

### Blockquote with attribution

#### Syntax

```markdown
> Don't communicate by sharing memory, share memory by communicating.<br>
> — <cite>Rob Pike[^1]</cite>
```

#### Output

> Don't communicate by sharing memory, share memory by communicating.<br>
> — Rob Pike[^1]

### Citations/Footnotes

#### Syntax

```markdown
[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.
```

#### Output

See bottom of page.

[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.

## Tables

### Syntax

```markdown
| Italics   | Bold     | Code   |
| --------- | -------- | ------ |
| _italics_ | **bold** | `code` |
```

### Output

| Italics   | Bold     | Code   |
| --------- | -------- | ------ |
| _italics_ | **bold** | `code` |

## Colors

### Syntax

```markdown
- <span font-bold font-mono text-amber>MAJOR</span>: Increment when you make incompatible API changes.
- <span font-bold font-mono text-lime>MINOR</span>: Increment when you add functionality in a backwards-compatible manner.
- <span font-bold font-mono text-blue>PATCH</span>: Increment when you make backwards-compatible bug fixes.
```

### Output

- <span font-bold font-mono text-amber>MAJOR</span>: Increment when you make incompatible API changes.
- <span font-bold font-mono text-lime>MINOR</span>: Increment when you add functionality in a backwards-compatible manner.
- <span font-bold font-mono text-blue>PATCH</span>: Increment when you make backwards-compatible bug fixes.

## Math/Equation Rendering

We use MathJax for rendering LaTeX. Here are a few examples of algorithms and equations:

### [Merge sort](https://en.wikipedia.org/wiki/Merge_sort)

#### Syntax

```markdown
The time complexity of merge sort is $O(n \log n)$.

The recursive relation is:

$$
T(n) = \begin{cases}
1 & \text{if } n = 1 \\
2T(n/2) + n & \text{if } n > 1
\end{cases}
$$

Big-O notation: $\textcolor{cyan}{f(n)} = O(\textcolor{magenta}{g(n)})$ means $\exists \textcolor{red}{c}, n_0$ such that $\textcolor{cyan}{f(n)} \leq \textcolor{red}{c} \cdot \textcolor{magenta}{g(n)}$ for all $n \geq n_0$.
```

#### Output

The time complexity of merge sort is $O(n \log n)$.

The recursive relation is:

$$
T(n) = \begin{cases}
1 & \text{if } n = 1 \\
2T(n/2) + n & \text{if } n > 1
\end{cases}
$$

Big-O notation: $\textcolor{cyan}{f(n)} = O(\textcolor{magenta}{g(n)})$ means $\exists \textcolor{red}{c}, n_0$ such that $\textcolor{cyan}{f(n)} \leq \textcolor{red}{c} \cdot \textcolor{magenta}{g(n)}$ for all $n \geq n_0$.

### Maxwell's Equations

#### Syntax

```markdown
In table form:
| equation | description |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| $\nabla \cdot \vec{\mathbf{B}}  = 0$ | divergence of $\vec{\mathbf{B}}$ is zero |
| $\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t}  = \vec{\mathbf{0}}$ | curl of $\vec{\mathbf{E}}$ is proportional to the rate of change of $\vec{\mathbf{B}}$ |
| $\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} = \frac{4\pi}{c}\vec{\mathbf{j}}    \nabla \cdot \vec{\mathbf{E}} = 4 \pi \rho$ | _???_ |

In array form:

$$
\begin{array}{c}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} &
= \frac{4\pi}{c}\vec{\mathbf{j}}    \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} & = 0
\end{array}
$$
```

#### Output

In table form:
| equation | description |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| $\nabla \cdot \vec{\mathbf{B}}  = 0$ | divergence of $\vec{\mathbf{B}}$ is zero |
| $\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t}  = \vec{\mathbf{0}}$ | curl of $\vec{\mathbf{E}}$ is proportional to the rate of change of $\vec{\mathbf{B}}$ |
| $\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} = \frac{4\pi}{c}\vec{\mathbf{j}}    \nabla \cdot \vec{\mathbf{E}} = 4 \pi \rho$ | _???_ |

In array form:

$$
\begin{array}{c}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} &
= \frac{4\pi}{c}\vec{\mathbf{j}}    \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} & = 0
\end{array}
$$

### Homomorphism

#### Syntax

```markdown
A homomorphism is a map between two algebraic structures of the same type (that is of the same name), that preserves the operations of the structures. This means a map $f:A \to B$ between two sets $A$, $B$ equipped with the same structure such that, if $\cdot$ is an operation of the structure (supposed here, for simplification, to be a binary operation), then

$$
\begin{equation}
f(x\cdot y)=f(x)\cdot f(y)
\end{equation}
$$

for every pair $x$, $y$ of element of $A$. One says often that $f$ preserves the operation or is compatible with the operation.

Formally, a map $f:A \to B$ preserves an operation $\mu$ of arity $\mathsf{k}$, defined on both $A$ and $B$ if

$$
\begin{equation}
f(\mu_A(a_1,\ldots,a_k))=\mu_B(f(a_1),\ldots,f(a_k))
\end{equation}
$$

for all elements $a_1,\ldots,a_k$ in $A$.
```

#### Output

A homomorphism is a map between two algebraic structures of the same type (that is of the same name), that preserves the operations of the structures. This means a map $f:A \to B$ between two sets $A$, $B$ equipped with the same structure such that, if $\cdot$ is an operation of the structure (supposed here, for simplification, to be a binary operation), then

$$
\begin{equation}
f(x\cdot y)=f(x)\cdot f(y)
\end{equation}
$$

for every pair $x$, $y$ of element of $A$. One says often that $f$ preserves the operation or is compatible with the operation.

Formally, a map $f:A \to B$ preserves an operation $\mu$ of arity $\mathsf{k}$, defined on both $A$ and $B$ if

$$
\begin{equation}
f(\mu_A(a_1,\ldots,a_k))=\mu_B(f(a_1),\ldots,f(a_k))
\end{equation}
$$

for all elements $a_1,\ldots,a_k$ in $A$.

## Code Blocks

### Syntax

We can use 3 backticks <code>```</code> on a new line, write our code snippet, and then close with 3 backticks on another new line. To highlight language specific syntax, we can type the language name after the first 3 backticks (e.g. `html`, `javascript`, `css`, `markdown`, `typescript`, `txt`, `bash`, `python`, etc).

````markdown
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Example HTML5 Document</title>
  </head>
  <body>
    <p>Test</p>
  </body>
</html>
```
````

### Output

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Example HTML5 Document</title>
  </head>
  <body>
    <p>Test</p>
  </body>
</html>
```

## More Code Renders

### Syntax

> [!NOTE]
> Only some transformers have been configured, see https://shiki.style/packages/transformers for more!

### Output

```ts
// transformerNotationDiff
console.log('hewwo') // [!code --]
console.log('hello') // [!code ++]
console.log('goodbye')
```

```python
# transformerNotationHighlight
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit([[1], [2], [3]], [2, 4, 6]) # [!code hl]
print(model.predict([[4]]))  # Output: [8]
```

```python
# transformerNotationWordHighlight
# [!code word:RandomForestClassifier]
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
print(RandomForestClassifier().fit(X, y).score(X, y))  # ~0.97
```

```python
# transformerNotationFocus
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True) # [!code focus]
print(RandomForestClassifier().fit(X, y).score(X, y))  # ~0.97
```

```ts
// transformerNotationErrorLevel
console.log('No errors or warnings')
console.error('Error') // [!code error]
console.warn('Warning') // [!code warning]
```

```js {2,4-5}
// transformerMetaHighlight
console.log('2')
console.log('3')
console.log('4')
console.log('5')
```

```js /Hello/
// transformerMetaWordHighlight
const msg = 'Hello World'
console.log(msg)
console.log(msg) // prints Hello World
```

## List Types

### Ordered List

#### Syntax

```markdown
1. First item
2. Second item
3. Third item
```

#### Output

1. First item
2. Second item
3. Third item

### Unordered List

#### Syntax

```markdown
- List item
- Another item
- And another item
```

#### Output

- List item
- Another item
- And another item

### Nested list

#### Syntax

```markdown
- Fruit
  - Apple
  - Orange
  - Banana
- Dairy
  - Milk
  - Cheese
```

#### Output

- Fruit
  - Apple
  - Orange
  - Banana
- Dairy
  - Milk
  - Cheese

## Other Elements — abbr, sub, sup, kbd, mark

### Syntax

```markdown
<abbr title="Graphics Interchange Format">GIF</abbr> is a bitmap image format.

H<sub>2</sub>O

X<sup>n</sup> + Y<sup>n</sup> = Z<sup>n</sup>

Press <kbd>CTRL</kbd> + <kbd>ALT</kbd> + <kbd>Delete</kbd> to end the session.

Most <mark>salamanders</mark> are nocturnal, and hunt for insects, worms, and other small creatures.
```

### Output

<abbr title="Graphics Interchange Format">GIF</abbr> is a bitmap image format.

H<sub>2</sub>O

X<sup>n</sup> + Y<sup>n</sup> = Z<sup>n</sup>

Press <kbd>CTRL</kbd> + <kbd>ALT</kbd> + <kbd>Delete</kbd> to end the session.

Most <mark>salamanders</mark> are nocturnal, and hunt for insects, worms, and other small creatures.

## Testing out Marimo

<iframe
  src="https://marimo.app/l/9bnuyz?embed=true&show-chrome=false"
  width="100%"
  height="300"
  frameborder="0"
></iframe>
