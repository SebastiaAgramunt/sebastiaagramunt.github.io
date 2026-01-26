---
title: Miller-Rabin primality test
author: sebastia
date: 2025-10-08 20:10:00 +0800
categories: [Cryptography]
tags: [cryptography, mathematics]
pin: true
toc: true
render_with_liquid: false
math: true
---

In the blog post - <a href="../prime-numbers">Prime Numbers</a> we have explained what is a primality test and specifically, the Miller-Rabin primality test. In this post we will implement a prototype of it in Python. Please, do not use it for production, it's just a dummy code to understand the Miller-Rabin algoirthm better. Before coding the final algorithm we need some other ingredients, we will explain from the most basic to the final algorithm.

## Exteneded Euclidean Algorithm

The greatest common divisor (GCD) of two numbers $a$ and $b$ is the larger number that divides both $a$ and $b$ leaving a reminder of zero. For instance, what is the greatest common divisor of 10 and 45?. What we would do is to decompose in it's prime numbers both numbers: $10=5 \cdot 2$, $45=3 \cdot 5 \cdot 3$. The largest common divisor is $5$. The [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) can find the greatest common divisor of two integer nubmers:

Let $a$ and $b \in \mathbb{Z}$ with $a \geq b \gt 0$. To find the greatest common divisor for $a$ and $b$ we begin by setting two integers

$$
r_{-1} = a,\qquad r_0 = b.
$$

Then for $k \geq 0$ we apply the division algorithm

$$
r_{k-1} = q_{k} \cdot r_{k} + r_{k+1}, \qquad 0 \leq r_{k+1} \lt r_k
$$

where $q_k \in \mathbb{Z}$ is the quotient and $r_{k+1}$ the remainder. We keep on calculating for $k=0, 1, 2, \cdots$ until $r_{k+1}$ is zero, then the greatest common divisor of $a$ and $b$ will be the previous reminder $r_k$. 

Let's apply the euclidean algorithm to find the greatest common divisor of 1067 and 462. We begin with $r_{-1}=1067$ and $r_0=462$, this gives a quocient of $2$ and a remainder of $143$, since the remainder is different from zero we need to apply another iteration. We take the quocient we got before $462$ and the remainder $143$, we find $q_1=3$ and $r_1=33$, since $r_1 \neq 0$ we iterate again... finally at $r_3$ we get zero, therefore $r_2$ is the greatest common divisor of $1067$ and $462$.

For better understanding I show all the steps in the following

$$
\begin{aligned}
a &= 1067,\ b = 462 
&\rightarrow&\ 1067 = q_0 \times 462 + r_0 
&\rightarrow&\ q_0 = 2,\ r_0 = 143 \\

a &= 462,\ b = 143 
&\rightarrow&\ 462 = q_1 \times 143 + r_1 
&\rightarrow&\ q_1 = 3,\ r_1 = 33 \\

a &= 143,\ b = 33 
&\rightarrow&\ 143 = q_2 \times 33 + r_2 
&\rightarrow&\ q_2 = 4,\ r_2 = 11 \\

a &= 33,\ b = 11
&\rightarrow&\ 33 = q_3 \times 11 + r_3
&\rightarrow&\ q_3 = 3,\ r_3 = 0
\end{aligned}
$$

giving

$$
\gcd(1067,462)=11
$$

Another way to calculate the greatest common divisor is the algorithm for the [extended euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm) and this is the one we will use for Miller-Rabin primality testing as it gives more information than the GCD. In the book **An Introduction to Mathematical Cryptography** by Hoffstein, Pipher and Silverman we can find the theorem that describes the extended Euclidean algorithm:

Let $a$ and $b$ be positive integers. Then the equation

$$
au + bv = \textit{gcd}(a, b)
$$

always has a solution in integers $u$ and $v$. If $(u_0, v_0)$ is any one solution, then every solution has the form

$$
u = u_0 + \frac{b \times k}{\textit{gcd}(a,b)}, \qquad v = v_0 - \frac{a \times k}{\textit{gcd}(a,b)}
$$

for some $k \in \mathbb{Z}$. We won't dive into the details of the proof for this theorem but you can check them in the book. The extended euclidean algorithm will proof very useful to find inveres in modulo operations and specifically using the Chinese reminder theorem that we will explain later.

The key idea behind the extended euclidean algorithm is that every reminer produced by the euclidean algorithm can be written as

$$
r_k = u_k \cdot a + v_k \cdot b
$$

The extended algorithm maintains these coefficients as the reminders evolve. When the algorithm ends, the last non zero remainder is the gcd, and its coefficients give the desired identity. Initially we start with

$$
r_{-1}=a \qquad r_0=b
$$


```python
def xgcd(a: int, b: int) -> tuple[int, int, int]:
    u0, u1, v0, v1 = 0, 1, 1, 0

    while a != 0:
        q, b, a = b // a, a, b % a
        v0, v1 = v1, v0 - q * v1
        u0, u1 = u1, u0 - q * u1

    return b, u0, v0
```