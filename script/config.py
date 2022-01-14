import configparser

CONFIG=configparser.ConfigParser()
CONFIG.optionxform=str
CONFIG.read("../config/config.conf")

if __name__=='__main__':
    print(CONFIG["Dataset"])
