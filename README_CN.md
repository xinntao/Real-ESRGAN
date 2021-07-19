# ProjectTemplate-Python

[English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/ProjectTemplate-Python) **|** [Gitee码云](https://gitee.com/xinntao/ProjectTemplate-Python)

## 文件修改

1. 设置 *pre-commit* hook.
    1. 若需要, 修改 `.pre-commit-config.yaml`
    1. 在文件夹根目录, 运行
    > pre-commit install
1. 修改 `.gitignore` 文件
1. 修改 `LICENSE` 文件
    本仓库使用 *MIT* 许可, 根据需要可以修改成其他许可
1. 修改 *setup* 文件
    1. `setup.cfg`
    1. `setup.py`, 特别是其中包含的关键字 `basicsr`
1. 修改 `requirements.txt` 文件
1. 修改 `VERSION` 文件

## GitHub Workflows

1. [pylint](./github/workflows/pylint.yml)
1. [gitee-repo-mirror](./github/workflow/gitee-repo-mirror.yml) - 支持 Gitee码云
    1. 在 [Gitee](https://gitee.com/) 网站克隆 Github 仓库
    1. 修改 [gitee-repo-mirror](./github/workflow/gitee-repo-mirror.yml) 文件
    1. 在 Github 中的 *Settings* -> *Secrets* 的 `SSH_PRIVATE_KEY`

## 其他流程

1. 主页上的 `description`, `website`, `topics`
1. 支持中文文档, 比如, `README_CN.md`

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

## 有用的图像链接

<img src="https://colab.research.google.com/assets/colab-badge.svg" height="28" alt="google colab logo">  Google Colab Logo <br>
<img src="https://upload.wikimedia.org/wikipedia/commons/8/8d/Windows_darkblue_2012.svg" height="28" alt="google colab logo">  Windows Logo <br>
<img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Logo-ubuntu_no%28r%29-black_orange-hex.svg" alt="Ubuntu" height="24">  Ubuntu Logo <br>

## 其他有用的技巧

1. `More` 下拉菜单
    <details>
    <summary>More</summary>
    <ul>
    <li>Nov 19, 2020. Set up ProjectTemplate-Python.</li>
    </ul>
    </details>
