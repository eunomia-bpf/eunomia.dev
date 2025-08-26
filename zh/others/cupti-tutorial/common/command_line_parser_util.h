#pragma once

#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>

class CommandLineParser
{
    struct OptionBase
    {
        virtual ~OptionBase() = default;
        virtual void parseValue(const std::string& str) = 0;
        virtual std::string getTypeName() const = 0;
        virtual std::string getDefaultValueStr() const = 0;
        std::string description;
        bool isFlag = false;
    };

    template<typename T>
    struct Option : OptionBase
    {
        T m_value;
        T m_defaultValue;

        Option(const T& defaultValue, const std::string& desc)
        {
            m_value = defaultValue;
            this->m_defaultValue = defaultValue;
            this->description = desc;
            if (std::is_same<T, bool>::value) {
                this->isFlag = true;
            }
        }

        void parseValue(const std::string& str) override
        {
            std::istringstream ss(str);
            ss >> std::boolalpha >> m_value;
            if (ss.fail()) {
                throw std::runtime_error("Failed to parse: " + str);
            }
        }

        std::string getTypeName() const override
        {
            if (std::is_same<T, int>::value) {
                return "int";
            } else if (std::is_same<T, double>::value) {
                return "double";
            } else if (std::is_same<T, bool>::value) {
                return "flag";
            } else {
                return "string";
            }
        }

        std::string getDefaultValueStr() const override
        {
            std::ostringstream ss;
            ss << std::boolalpha << m_defaultValue;
            return ss.str();
        }
    };

    std::unordered_map<std::string, std::string> m_aliasMap;  // short -> long
    std::map<std::string, std::unique_ptr<OptionBase>> m_optionsMap;

public:
    template<typename T>
    void addOption(const std::string& shortName, const std::string& longName, const std::string& desc, const T& defaultValue)
    {
        if (m_optionsMap.count(longName) || (!shortName.empty() && m_optionsMap.count(shortName))) {
            throw std::runtime_error("Duplicate option: " + longName + " or " + shortName);
        }

        auto opt = std::unique_ptr<OptionBase>(new Option<T>(defaultValue, desc));
        m_optionsMap[longName] = std::move(opt);

        if (!shortName.empty()) {
            m_aliasMap[shortName] = longName;
        }
    }

    void parse(int argc, char* argv[])
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h")
            {
                printHelp();
                exit(0);
            }

            if (m_aliasMap.count(arg)) {
                arg = m_aliasMap[arg];
            }

            auto it = m_optionsMap.find(arg);
            if (it == m_optionsMap.end()) {
                continue; // ignore unknown options
            }

            auto& opt = it->second;

            if (opt->isFlag) {
                opt->parseValue("true"); // flags like --verbose
            } else {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Expected value after " + arg);
                }
                opt->parseValue(argv[++i]);
            }
        }
    }

    template<typename T>
    T get(const std::string& name) const
    {
        auto it = m_optionsMap.find(name);
        if (it == m_optionsMap.end())
        {
            // return default value if not found
            auto aliasIt = m_aliasMap.find(name);
            if (aliasIt != m_aliasMap.end()) {
                it = m_optionsMap.find(aliasIt->second);
            } else {
                throw std::runtime_error("Unknown option: " + name);
            }

            if (it == m_optionsMap.end()) {
                return dynamic_cast<Option<T>*>(it->second.get())->m_defaultValue;
            }
        }

        auto opt = dynamic_cast<Option<T>*>(it->second.get());
        if (!opt) {
            throw std::runtime_error("Type mismatch on get for: " + name);
        }

        return opt->m_value;
    }

    void printHelp() const
    {
        std::cout << "Available options:\n\n";
        for (const auto& option : m_optionsMap)
        {
            std::string shortName;
            for (const auto& alias : m_aliasMap)
            {
                if (alias.second == option.first) {
                    shortName = alias.first;
                    break;
                }
            }

            std::ostringstream line;
            line << "  ";
            if (!shortName.empty()) {
                line << shortName << ", ";
            }
            line << option.first;

            std::cout << std::left << std::setw(40) << line.str();
            std::cout << option.second->description;

            std::cout << " [" << option.second->getTypeName()
                      << ", default=" << (option.second->getDefaultValueStr().empty() ? "<empty>" : option.second->getDefaultValueStr()) << "]\n";
        }
        std::cout << "\nUse --help or -h to display this message.\n";
    }
};
