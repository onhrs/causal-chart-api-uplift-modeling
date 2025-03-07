import matplotlib.pyplot as plt
import matplotlib
import os
import platform
import logging

logger = logging.getLogger(__name__)

def setup_japanese_fonts():
    """
    matplotlibで日本語フォントを使用するための設定
    """
    try:
        # システム別の日本語フォント設定
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            font_list = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'AppleGothic']
        elif system == 'Windows':
            font_list = ['MS Gothic', 'Yu Gothic', 'Meiryo']
        else:  # Linux など
            font_list = ['IPAGothic', 'Noto Sans CJK JP', 'TakaoGothic']
        
        # 利用可能なフォントを確認
        available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
        logger.info(f"Available fonts: {available_fonts}")
        
        # フォントリストから最初に見つかった利用可能なフォントを使用
        for font in font_list:
            if any(font.lower() in f.lower() for f in available_fonts):
                logger.info(f"Using font: {font}")
                plt.rcParams['font.family'] = font
                return True
        
        # 上記のフォントが見つからない場合はフォールバック
        logger.warning("日本語フォントが見つかりませんでした。代替手段を試みます。")
        
        # fontconfigを使ったフォールバック
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['IPAGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
        return True
        
    except Exception as e:
        logger.error(f"日本語フォント設定中にエラーが発生しました: {str(e)}")
        return False

def set_matplotlib_japanize():
    """
    日本語対応のためのmatplotlib設定を行う関数
    これは実行時にインポートするため、japanize-matplotlibがインストールされていない場合でもエラーにならない
    """
    try:
        import japanize_matplotlib
        logger.info("japanize-matplotlibを使用して日本語フォントを設定しました")
        return True
    except ImportError:
        logger.info("japanize-matplotlibが見つかりませんでした。代替方法を試みます。")
        return setup_japanese_fonts()
