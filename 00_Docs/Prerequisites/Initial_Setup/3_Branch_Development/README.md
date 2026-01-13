# 🔀 Software Engineering Using Branch Development

Classically, to use branch development, we’ll need to zoom in on two concepts we just mentioned above: [git branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) and [local repos vs. remotes](https://git-scm.com/book/ms/v2/Git-Basics-Working-with-Remotes). We will also need to understand Git commands and all that noise.

For now, though, forget memorizing Git commands. They will come naturally if we can understand the process at a higher level.

**TL;DR**

- General Flow: Upstream → Origin → Your Laptop (**Local**) → Origin
- The `main` branch in your remote repo stays synced with **Upstream** (course materials) *while upstream updates never overwrite your work*
- Feature (assignment) branches are used to do work on Local and to push work to **Origin** (your remote on GitHub)

Let’s boil it all down to 🎋 The Three Main Branch Roles:

## 🎋 The Three Main Branch Roles

1. `main`
    - “The current official story.”
    - In class: the latest course materials.
    - In industry: what’s in **production**.
    - *Deployed to users, only stable, released code.*
2. `develop` (professional Gitflow)
    - “The draft of the next chapter.”
    - In class: **Not Applicable**
    - In industry: what’s in **staging**.
    - *Where all features get integrated (feature branches get merged) before release*.
- `feature / assignment / release` **branches**
    - “Write and edit paragraphs before they go into the book.”
    - In class: e.g., `lesson1-assignment`
    - In industry: e.g., `feature/login-page` or `release/1.2.0`

**TL;DR**

- Don’t work directly on `main`.
- *Branch off* `main` (as in class) or `develop` (as in industry), do work, then merge back.

## 🏡 The Three Places Your Code Lives

For class, we don’t need to worry about having a `develop` or **staging** area. 

At the same time, 

- **Upstream**
    - The *source of truth* you don’t directly control (e.g., course materials).
- **Origin**
    - *Your* copy of the repo on GitHub (your remote).
- **Local**
    - The repo on your machine where you actually edit files (e.g., do assignments)

**TL;DR**

- You will only `pull` from **Upstream** (course materials), not `push` back to it.
- You will write code and edit files on **Local** (with assignment/feature branches).
- You will `push` to **Origin** (your remote repository).


### Thinking Questions

- What is one lesson you've learned from this?
- What is one [lesson that you have not yet learned](https://www.loom.com/share/b34e4bd657f74892ac9a01f774113b4d)?
