import React, { createContext, useContext, useState, useMemo, useEffect } from 'react';
import { 
    View, Text, StyleSheet, TouchableOpacity, SafeAreaView, ScrollView, 
    Switch, useColorScheme 
} from 'react-native';

// =================================================================
// --- THEME CONTEXT AND LOGIC (Required for global state) ---
// =================================================================

// --- 1. THEME COLORS ---
const LIGHT_COLORS = {
    accent: '#005A9C', 
    containerBg: '#FFFFFF', 
    screenBg: '#F0F9FF', 
    text: '#333', 
    muted: '#E0E0E0', 
    darkContrast: '#005A9C', 
    lightContrast: '#FFFFFF', 
    error: '#D32F2F', 
};

const DARK_COLORS = {
    accent: '#C8EAF7', 
    containerBg: '#003B6D', 
    screenBg: '#121212', 
    text: '#C8EAF7', 
    muted: '#005A9C', 
    darkContrast: '#005A9C',
    lightContrast: '#FFFFFF', 
    error: '#FF7F7F', 
};

// --- 2. PLACEHOLDER UTILITIES ---
const MaterialIcons = ({ name, size, color }) => <Text style={{ fontSize: size, color, marginRight: name === 'chevron-right' ? 0 : 5 }}>{name}</Text>; 
const useRouter = () => ({ back: () => console.log('Navigating back...') }); 
const loadThemePreference = async () => 'system'; 
const saveThemePreference = async (themeMode) => console.log(`[Persistence]: Theme preference saved: ${themeMode}`);


// --- 3. THEME CONTEXT AND CUSTOM HOOK ---
const ThemeContext = createContext({
    themeMode: 'system',
    colors: LIGHT_COLORS,
    setThemeMode: () => {},
});
const useTheme = () => useContext(ThemeContext);


// --- 4. THEME PROVIDER COMPONENT ---
const ThemeProvider = ({ children }) => {
    const [themePreference, setThemePreference] = useState('system'); 
    const [isLoaded, setIsLoaded] = useState(false); 
    const systemTheme = useColorScheme();

    useEffect(() => {
        const loadSavedTheme = async () => {
            const savedTheme = await loadThemePreference();
            if (savedTheme) {
                setThemePreference(savedTheme);
            }
            setIsLoaded(true);
        };
        loadSavedTheme();
    }, []);

    const handleSetThemeMode = (mode) => {
        setThemePreference(mode);
        saveThemePreference(mode);
    };

    const colors = useMemo(() => {
        let activeMode = themePreference;
        if (themePreference === 'system') {
            activeMode = systemTheme === 'dark' ? 'dark' : 'light';
        }
        return activeMode === 'dark' ? DARK_COLORS : LIGHT_COLORS;
    }, [themePreference, systemTheme]);


    const contextValue = useMemo(() => ({
        themeMode: themePreference,
        colors,
        setThemeMode: handleSetThemeMode,
    }), [themePreference, colors]);

    if (!isLoaded) {
        return <View style={{ flex: 1, backgroundColor: LIGHT_COLORS.screenBg }} />;
    }

    return (
        <ThemeContext.Provider value={contextValue}>
            {children}
        </ThemeContext.Provider>
    );
};


// =================================================================
// --- UI COMPONENTS ---
// =================================================================

// Reusable component for a segmented selection (Theme Selector)
const SegmentedSelector = ({ options, selected, onSelect, themeColors }) => (
    <View style={segmentedStyles(themeColors).container}>
        {options.map((option) => (
            <TouchableOpacity
                key={option.value}
                style={[
                    segmentedStyles(themeColors).option,
                    selected === option.value && segmentedStyles(themeColors).selectedOption,
                ]}
                onPress={() => onSelect(option.value)}
            >
                <Text
                    style={[
                        segmentedStyles(themeColors).optionText,
                        selected === option.value && segmentedStyles(themeColors).selectedOptionText,
                    ]}
                >
                    {option.label}
                </Text>
            </TouchableOpacity>
        ))}
    </View>
);

// Setting Item Component for presentation
const SettingItem = ({ title, component, themeColors }) => (
    <View style={getStyles(themeColors).settingItemWrapper}>
        <View style={getStyles(themeColors).settingItem}>
            <Text style={getStyles(themeColors).settingText}>{title}</Text>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                {component}
            </View>
        </View>
    </View>
);


// --- THE APP PREFERENCES SCREEN ---
export function AppPreferencesScreen() {
    // Access the global theme state and setter
    const { colors, themeMode, setThemeMode } = useTheme();
    const router = useRouter();
    const dynamicStyles = getStyles(colors);

    // Placeholder for other preferences
    const [autoDownloadEnabled, setAutoDownloadEnabled] = useState(true);

    const themeOptions = [
        { label: "Light", value: 'light' },
        { label: "Dark", value: 'dark' },
        { label: "System", value: 'system' },
    ];

    return (
        <SafeAreaView style={dynamicStyles.container}>
            <View style={dynamicStyles.header}>
                <TouchableOpacity onPress={() => router.back()}>
                    <MaterialIcons name="arrow-back-ios" size={24} color={colors.accent} />
                </TouchableOpacity>
                <Text style={dynamicStyles.headerTitle}>App Preferences</Text>
                <View style={{ width: 24 }} />
            </View>
            
            <ScrollView contentContainerStyle={dynamicStyles.scrollContent}>
                
                {/* 1. APPEARANCE SECTION (The main reason for this screen) */}
                <View style={dynamicStyles.settingsContainer}>
                    <Text style={dynamicStyles.sectionTitle}>Appearance</Text>
                    
                    <SettingItem 
                        title="Application Theme" 
                        themeColors={colors}
                        component={
                            <SegmentedSelector
                                options={themeOptions}
                                selected={themeMode}
                                onSelect={setThemeMode} // This is the core logic: updates and saves theme
                                themeColors={colors}
                            />
                        }
                    />
                </View>

                {/* 2. EXAMPLE UTILITY SECTION (For context) */}
                <View style={dynamicStyles.settingsContainer}>
                    <Text style={dynamicStyles.sectionTitle}>Data Management</Text>
                    
                    <SettingItem 
                        title="Auto-Download Content (Wi-Fi Only)" 
                        themeColors={colors}
                        component={
                            <Switch
                                trackColor={{ false: colors.muted, true: colors.accent }}
                                thumbColor={colors.containerBg}
                                onValueChange={setAutoDownloadEnabled}
                                value={autoDownloadEnabled}
                            />
                        }
                    />

                    <TouchableOpacity 
                        style={[dynamicStyles.settingItem, { borderBottomWidth: 0 }]}
                        onPress={() => console.log('Tapped Language')}
                    >
                        <Text style={dynamicStyles.settingText}>Language</Text>
                        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                            <Text style={{ color: colors.muted, fontSize: 16 }}>English (US)</Text>
                            <MaterialIcons name="chevron-right" size={24} color={colors.muted} />
                        </View>
                    </TouchableOpacity>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
};

// --- MAIN APP WRAPPER (Renders the App Preferences screen) ---
export default function App() {
    return (
        <ThemeProvider>
            <AppPreferencesScreen /> 
        </ThemeProvider>
    );
}


// --- STYLING FUNCTIONS ---

const segmentedStyles = (colors) => StyleSheet.create({
    container: {
        flexDirection: 'row',
        backgroundColor: colors.muted,
        borderRadius: 12,
        overflow: 'hidden',
        height: 40,
        width: 200, 
        padding: 4,
    },
    option: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    selectedOption: {
        backgroundColor: colors.darkContrast,
        borderRadius: 8,
    },
    optionText: {
        fontSize: 14,
        fontWeight: '500',
        color: colors.darkContrast,
    },
    selectedOptionText: {
        color: colors.lightContrast,
        fontWeight: '700',
    }
});


const getStyles = (colors) => StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.screenBg,
    },
    header: {
        height: 60,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        backgroundColor: colors.containerBg, 
        borderBottomWidth: 1,
        borderBottomColor: colors.muted,
        elevation: 4,
    },
    headerTitle: {
        fontSize: 22,
        fontWeight: '800',
        color: colors.accent,
    },
    scrollContent: {
        paddingVertical: 20,
        flex: 1,
    },
    settingsContainer: {
        marginHorizontal: 15,
        marginBottom: 20,
        backgroundColor: colors.containerBg,
        borderRadius: 18,
        paddingHorizontal: 20,
        paddingVertical: 10,
        elevation: 5,
    },
    sectionTitle: {
        fontSize: 17,
        fontWeight: '700',
        color: colors.accent,
        marginTop: 5,
        marginBottom: 5,
        borderBottomWidth: 1,
        borderBottomColor: colors.muted,
        paddingBottom: 10,
        letterSpacing: 0.5,
        textTransform: 'uppercase',
    },
    settingItemWrapper: {
        borderBottomWidth: 1,
        borderBottomColor: colors.muted,
    },
    settingItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: 10,
        minHeight: 50,
    },
    settingText: {
        fontSize: 16,
        color: colors.text,
        fontWeight: '500',
        flexShrink: 1,
        marginRight: 10,
    },
});
