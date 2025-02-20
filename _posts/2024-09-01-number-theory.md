---
title: Number theory for cryptography and privacy preserving machine learning
author: sebastia
date: 2024-09-01 20:08:00 +0800
categories: [Cryptography]
tags: [cryptography, mathematics]
pin: true
toc: true
render_with_liquid: false
math: true
---

This is a first post in which I intend to explain the basic ingredients needed to understand the cryptography for privacy preserving machine learning topics. Here I will cover number theory, for python code check the [github repository](https://github.com/SebastiaAgramunt/Cryptography).

## Introduction

In this post I will focus on the most basic ingredients for cryptography, basic number theory. This is needed to understand all sorts of cryptograhpt: symmetric vs asymmetric cryptography, hash functions, digital signatures, random number generation, key exchange protocols, secret sharing schemes, homomorphisms and secure computation among others.


## Divisibility and greatest common divisor

In this section I write about division of integer numbers. This will be used through all this post in one way or another. Given two natural numbers a and b (the latter nonzero) we say that b divides a if there is another integer $c$ such that $a=b \cdot c$. If, conversely we can’t find such $c$ then if $b<a$ we can find a relation $a=b \cdot q+r$ where $q$ is called quotient and $r$ remainder. This, in other words is what we learn at primary school, just a bit more formalised.

If there’s a number $d$ that divides both, $a$ and $b$, we also say that $d$ is a common divisor of $a$ and $b$. For instance, 2 is a common divisor of 60 and 80 since you can write 60=30*2 and 80=40*2. One way to calculate the greatest common divisor (gcd) of two integers is through the extended euclidean algorithm: given two positive integers $a$ and $b$, the following equation holds

$$
au+bv=\textit{gcd}(a,b)
$$

The python code to solve this equation can be found here. For instance, the solution for the pair $(a, b)=(30,27)$ is $g, u, v = (3, 1, -1)$ where $g$ is the gcd. And a last definition, $a$ and $b$ are said to be **coprime** iff $gcd(a, b)=1$, that is the largest number that divides both is 1.

## Modular arithmetic

Modular arithmetic is a system of arithmetic for integers, where numbers “wrap around” when reaching a certain value. First we need to fix a value m and then compute the modulo over an integer $a$, the result is the remainder of the division. For instance, if $m=7$

| i | i(mod 7) |
|---|----------|
| 0 | 0        |
| 1 | 1        |
| 2 | 2        |
| 3 | 3        |
| 4 | 4        |
| 5 | 5        |
| 6 | 6        |
| 7 | 0        |
| 8 | 1        |
| 9 | 2        |

you can see that the value of the operation remains the same for $i<m$, when it reaches m it “wraps around” and begins with 0 again.

Now you may wonder… why does this have to do with cryptography? Well, there are two reasons, the first and most important is that modular arithmetic allows the construction of simple algebras like groups or fields, these are the building blocks of cryptography. The second reason is that this defines a finite set of elements (not infinite like natural numbers and real numbers) and therefore is more tractable on a computer.

The modulo operation defines an [algebraic group](https://en.wikipedia.org/wiki/Group_(mathematics)) over the sum, so if we take the previous example of $m=7$, the elements of the group are $(0, 1, 2, 3, 4, 5, 6)$ and the operation is the sum modulo m. See the "multiplication" table for operation sum modulo:


| + | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| 1 | 1 | 2 | 3 | 4 | 5 | 6 | 0 |
| 2 | 2 | 3 | 4 | 5 | 6 | 0 | 1 |
| 3 | 3 | 4 | 5 | 6 | 0 | 1 | 2 |
| 4 | 4 | 5 | 6 | 0 | 1 | 2 | 3 |
| 5 | 5 | 6 | 0 | 1 | 2 | 3 | 4 |
| 6 | 6 | 0 | 1 | 2 | 3 | 4 | 5 |

A group has the following properties

* **Element closure**: for any two elements a and b of the group, their operation returns c which is also in the group. E.g. (5+3)(mod 7)=1
* **Associativity**: for any three elements a, b, c it holds (a+b)+c=a+(b+c). This obviously happens in the sum operation.
* **Existence of identity**: There exist an element e in the set such that for any a in the set a+e=a. In this example the neutral element is 0. E.g. (5+0)(mod 7)=5.
* **Inverse element**: For any element in the group a there must be another element b such that a+b=e. E.g. the inverse of 5 is 2 in our example because (5+2)(mod 7)=0

If the group is commutative (i.e. $a+b=b+a$), then the group is also called **commutative group** or **abelian**.


## Modulo operations on product

The sum modulo operation works well to construct a group, just choose an $m$ (any $m$ of the natural numbers will work) and you will be in the realm of the numbers $(0, 1, 2, …, m-1)$ and the modulo operation with addition. But what if instead of the sum we choose the product to define a group?. In this case the neutral element is $1$ and we’ll find out that sometimes we cannot form a group for arbitrary $m$, for instance take $m=10$, are you able to find the multiplicative inverse of 2, i.e. find $x$ such that

$$
2 \cdot x =1 \pmod{10}
$$

Let's inspect the entire multiplication table  for $2 \cdot x \pmod{10}$

| $x$ | $2 \cdot x \pmod{10}$  |
|-----|------------------------|
| 0   | 0                      |
| 1   | 2                      |
| 2   | 4                      |
| 3   | 6                      |
| 4   | 8                      |
| 5   | 0                      |
| 6   | 2                      |
| 7   | 4                      |
| 8   | 6                      |
| 9   | 8                      |


From the multiplication table you can see that there’s no number that multiplied by $2$ gives the neutral multiplication number $1$ in the field $\pmod{10}$. We can say that these elements do not form a group because we are missing inverse elements on some (if you check it, those without inverse are 2, 4, 5, 6, 8). But good news we can pick those elements that have inverse and form a group!. Let me write a multiplication table for such group in $m=10$:

| x | 1 | 3 | 7 | 9 |
|---|---|---|---|---|
| 1 | 1 | 3 | 7 | 9 |
| 3 | 3 | 9 | 1 | 7 |
| 7 | 7 | 1 | 9 | 3 |
| 9 | 9 | 7 | 3 | 1 |

See that all elements in this list ${1, 3, 7, 9}$ have an inverse. We can say that ${1, 3, 7, 9}$ with the operation multiplication $\pmod{10}$ form an abelian group.

Is there a way to know if an element has inverse modulo $m$? Yes!. Let $a$ and $m$ be integers such that $a<m$, then $a \cdot b \pmod{m}=1$ for some integer $b$ if and only if $gcd(a, m)=1$. Actually we can use the **extended euclidean algorithm** to calculate the inverse:

If $a$ has inverse modulo $m$, then from the stated above:

$$
au+mv=1 \pmod{m}
$$

by applying $\pmod{m}$ to both sides of the equation we will have

$$
au=1 \pmod{m}
$$

and therefore $u$ is the inverse of $a \pmod{m}$. There is a special case when $m$ is a prime number, we will denote a general prime number $p$ from now on. In this case, $gcd(a, p)=1$ since $p$ is only divisible by himself and $1$, therefore all the elements $(1, 2, 3, …, p-1)$ will have inverse and will form a group with the product $\pmod{p}$. An easy way to calculate the multiplicative inverse in a prime modulo group is to use the **Fermat’s little theorem**: Let $p$ be a prime number and let $a$ be any integer then:

$$
a^{p-1}=1 \pmod{p}
$$

if $a$ is not divisible by $p$, otherwise the above equation equals $0$. Then to calculate the inverse of $a$, we just need to multiply by $a^{-1}$ both sides of the equation:

$$
a^{-1}=a^{p-2}\pmod{p}
$$

so we can now calculate the inverse modulo $p$ of $a$. This may seem computationally expensive but we use the fast powering algorithm an implementation of which in Python can be found [here](https://github.com/SebastiaAgramunt/Cryptography/blob/master/notebooks/crypt.py#L135).


## Rings and Fields

So far we have seen how to to define mathematical groups with sum and multiplication modulo an integer $m$. We can construct other algebraic structures using both operations. A **ring** is an algebraic structure consisting of a set of elements $S$ and two operations $(+, \cdot)$ that fulfil the following properties

* The set with the first operation form an abelian group. i.e. $(S, +)$ is abelian
* There’s associativity on the second operation. I.e. if $a$, $b$, $c$ are elements of $S$ then $(a \cdot b) \cdot c=a \cdot (b \cdot c)$.
* Existence of identity on the second operation. I.e. there exist an element $e$ such that for any $a$ in the set $a \cdot e=a$.
* The second operation is distributive with respect to the first one. This is $a \cdot (b + c)=a \cdot b +a \cdot c$ and $(b + c) \cdot a = b \cdot a + c \cdot a$.

The set $S={0, 1, 2, 3}$ with the modular operations of additions ($+ \pmod{4})$ and multiplications ($\cdot \mod{4}$) is a ring. Another example of ring is the set of matrices of dimension $3 \times 3$ and real coefficients. You can check that all properties above are accomplished (takehome exercise!).

A **field** is a ring such that the second operation also satisfies all the abelian group properties (after throwing out the identity element of the first operation). The field has multiplicative inverses, multiplicative identity and is commutative.


A typical example of field is the set $S={0, 1, 2, 3, …, p}$ where $p$ is a prime number and operations are addition and multiplication modulo $p$. See that since $p$ is a prime all the elements of the set have a multiplicative inverse and therefore constitute an abelian group with the multiplication operation. This field is commonly denoted as $\mathbb{F}_p$. The set of matrices with dimension $3 \times 3$ and real coefficients is not a field since some matrices do not have a multiplicative inverse.


## Conclusions

We understood what is modulo arithmetic and how with such operation we can define groups, rings and fields over finite sets of elements. These structures appear constantly in cryptography, for instance when working with elliptic curves or in simple protocols like Diffie-Hellman key exchange.

In the next posts we are going to work with the defined algebraic structures to understand key concepts on cryptography and multiparty computation.

Thank you for reading!. If you like the article, please star my [github repository](https://github.com/SebastiaAgramunt/Cryptography).