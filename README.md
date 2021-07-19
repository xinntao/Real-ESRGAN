# ProjectTemplate-Python

[English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/ProjectTemplate-Python) **|** [Gitee码云](https://gitee.com/xinntao/ProjectTemplate-Python)

## File Modification

1. Setup *pre-commit* hook
    1. If necessary, modify `.pre-commit-config.yaml`
    1. In the repository root path, run
    > pre-commit install
1. Modify the `.gitignore` file
1. Modify the `LICENSE` file
    This repository uses the *MIT* license, you may change it to other licenses
1. Modify the *setup* files
    1. `setup.cfg`
    1. `setup.py`, especially the `basicsr` keyword
1. Modify the `requirements.txt` files
1. Modify the `VERSION` file

## GitHub Workflows

1. [pylint](./github/workflows/pylint.yml)
1. [gitee-repo-mirror](./github/workflow/gitee-repo-mirror.yml) - Support Gitee码云
    1. Clone GitHub repo in the [Gitee](https://gitee.com/) website
    1. Modify [gitee-repo-mirror](./github/workflow/gitee-repo-mirror.yml)
    1. In Github *Settings* -> *Secrets*, add `SSH_PRIVATE_KEY`

## Other Procedures

1. The `description`, `website`, `topics` in the main page
1. Support Chinese documents, for example, `README_CN.md`

## Emoji

[Emoji cheat-sheet](https://github.com/ikatyang/emoji-cheat-sheet)

| Emoji | Meaning |
| :---         |     :---:      |
| :rocket:   | Used for [BasicSR](https://github.com/xinntao/BasicSR) Logo |
| :sparkles: | Features |
| :zap: | HOWTOs |
| :wrench: | Installation / Usage |
| :hourglass_flowing_sand: | TODO list |
| :turtle: | Dataset preparation |
| :computer: | Commands |
| :european_castle: | Model zoo |
| :memo: | Designs |
| :scroll: | License and acknowledgement |
| :earth_asia: | Citations |
| :e-mail: | Contact |
| :m: | Models |
| :arrow_double_down: | Download |
| :file_folder: | Datasets |
| :chart_with_upwards_trend: | Curves|
| :eyes: | Screenshot |
| :books: |References |

## Useful Image Links

<img src="https://colab.research.google.com/assets/colab-badge.svg" height="28" alt="google colab logo">  Google Colab Logo <br>
<img src="https://upload.wikimedia.org/wikipedia/commons/8/8d/Windows_darkblue_2012.svg" height="28" alt="google colab logo">  Windows Logo <br>
<img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Logo-ubuntu_no%28r%29-black_orange-hex.svg" alt="Ubuntu" height="24">  Ubuntu Logo <br>

## Other Useful Tips

1. `More` drop-down menu
    <details>
    <summary>More</summary>
    <ul>
    <li>Nov 19, 2020. Set up ProjectTemplate-Python.</li>
    </ul>
    </details>
