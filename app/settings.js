import React from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    SafeAreaView,
    ScrollView,
    Alert,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';

export default function Settings() {
    const router = useRouter();

    // The handleLogout function is kept here as requested, 
    // even though the button is removed from the component's render.
    const handleLogout = async () => {
        try {
            // Confirmation dialog before logging out
            Alert.alert(
                "Log Out",
                "Are you sure you want to log out?",
                [
                    { text: "Cancel", style: "cancel" },
                    {
                        text: "Yes, Log Out",
                        onPress: async () => {
                            await SecureStore.deleteItemAsync('userToken');
                            await SecureStore.deleteItemAsync('userProfile');
                            // Replace history to prevent going back to protected screens
                            router.replace('/login');
                        },
                        style: "destructive"
                    }
                ]
            );

        } catch (error) {
            console.error('Logout error:', error);
            Alert.alert('Error', 'Failed to log out. Please try again.');
        }
    };

    // --- General Setting Navigation Functions ---
    const handleNotifications = () => {
        // Navigates to notif-settings.js (route /notif-settings)
        router.push('/notif-settings');
    };

    const handleAppPreferences = () => {
        // Navigates to app-preferences.js (route /app-preferences)
        router.push('/app-preferences');
    };

    // --- Account Setting Navigation Functions ---
    const handleChangePassword = () => {
        // Navigates to change-pass.js (route /change-pass)
        router.push('/change-pass');
    };

    const handleDeleteAccount = () => {
        // Navigates to delete-acc.js (route /delete-acc)
        router.push('/delete-acc');
    };


    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()}>
                    <MaterialIcons name="arrow-back-ios" size={24} color="#005A9C" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Settings</Text>
                <View style={{ width: 24 }} />
            </View>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                
                {/* General Settings Section */}
                <View style={styles.settingsContainer}>
                    <Text style={styles.sectionTitle}>General</Text>
                    
                    <TouchableOpacity 
                        style={styles.settingItem} 
                        onPress={handleNotifications}
                    >
                        <Text style={styles.settingText}>Notification Settings</Text>
                        <MaterialIcons name="chevron-right" size={24} color="#005A9C" />
                    </TouchableOpacity>

                    <TouchableOpacity 
                        style={[styles.settingItem, { borderBottomWidth: 0 }]} // Remove border from last item
                        onPress={handleAppPreferences}
                    >
                        <Text style={styles.settingText}>App Preferences</Text>
                        <MaterialIcons name="chevron-right" size={24} color="#005A9C" />
                    </TouchableOpacity>
                </View>

                {/* Account Settings Section */}
                <View style={styles.settingsContainer}>
                    <Text style={styles.sectionTitle}>Account</Text>

                    <TouchableOpacity 
                        style={styles.settingItem} 
                        onPress={handleChangePassword}
                    >
                        <Text style={styles.settingText}>Change Password</Text>
                        <MaterialIcons name="chevron-right" size={24} color="#005A9C" />
                    </TouchableOpacity>
                    
                    <TouchableOpacity 
                        style={[styles.settingItem, { borderBottomWidth: 0 }]} // Remove border from last item
                        onPress={handleDeleteAccount}
                    >
                        <Text style={[styles.settingText, { color: '#E53935' }]}>Delete Account</Text>
                        <MaterialIcons name="chevron-right" size={24} color="#005A9C" />
                    </TouchableOpacity>
                </View>

                {/* Removed the dedicated Logout Button UI block */}

            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#77CDE0'
    },
    header: {
        height: 56,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
    },
    headerTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: '#005A9C'
    },
    scrollContent: {
        paddingVertical: 20
    },
    settingsContainer: {
        marginHorizontal: 20,
        marginBottom: 20,
        backgroundColor: '#C8EAF7',
        borderRadius: 25,
        padding: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 10,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: '#005A9C',
        marginBottom: 15,
        borderBottomWidth: 1,
        borderBottomColor: '#B0DAE6',
        paddingBottom: 10,
    },
    settingItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: 15,
        borderBottomWidth: 1,
        borderBottomColor: '#B0DAE6',
    },
    settingText: {
        fontSize: 16,
        color: '#333',
    },
    // Removed unused logoutButton and logoutButtonText styles
});
