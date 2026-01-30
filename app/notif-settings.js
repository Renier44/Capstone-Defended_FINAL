import React, { useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    SafeAreaView,
    ScrollView,
    Switch,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

// Reusable component for a single notification setting row
const NotificationToggle = ({ label, value, onValueChange }) => (
    <View style={styles.settingItem}>
        <Text style={styles.settingText}>{label}</Text>
        <Switch
            trackColor={{ false: '#B0DAE6', true: '#005A9C' }}
            thumbColor={value ? '#C8EAF7' : '#f4f3f4'}
            ios_backgroundColor="#B0DAE6"
            onValueChange={onValueChange}
            value={value}
        />
    </View>
);

export default function NotificationSettings() {
    const router = useRouter();

    // State for managing different notification toggles
    const [isPushEnabled, setIsPushEnabled] = useState(true);
    const [isMessagesEnabled, setIsMessagesEnabled] = useState(true);
    const [isRemindersEnabled, setIsRemindersEnabled] = useState(false);
    const [isPromosEnabled, setIsPromosEnabled] = useState(false);

    // Handler for the master push notification toggle
    const handlePushToggle = (newValue) => {
        setIsPushEnabled(newValue);
        // If the main push is turned off, also turn off all sub-toggles (optional logic)
        if (!newValue) {
            setIsMessagesEnabled(false);
            setIsRemindersEnabled(false);
            setIsPromosEnabled(false);
        }
        // If the main push is turned on, restore sub-toggles to their last state (optional logic)
        // You might want to load persistent settings here instead of just toggling all.
    };

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()}>
                    <MaterialIcons name="arrow-back-ios" size={24} color="#005A9C" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Notification Settings</Text>
                <View style={{ width: 24 }} />
            </View>
            
            <ScrollView contentContainerStyle={styles.scrollContent}>
                
                {/* Master Push Notification Control */}
                <View style={styles.settingsContainer}>
                    <Text style={styles.sectionTitle}>Push Notifications</Text>
                    
                    <NotificationToggle
                        label="Enable All Push Notifications"
                        value={isPushEnabled}
                        onValueChange={handlePushToggle}
                    />

                    <Text style={[styles.infoText, { marginTop: 15 }]}>
                        Manage specific types of alerts below. These settings are controlled by the main switch above.
                    </Text>
                </View>
                
                {/* Specific Notification Categories */}
                <View style={styles.settingsContainer}>
                    <Text style={styles.sectionTitle}>Alert Categories</Text>

                    <NotificationToggle
                        label="Direct Messages"
                        value={isMessagesEnabled && isPushEnabled} // Controlled by master switch
                        onValueChange={setIsMessagesEnabled}
                    />
                    <NotificationToggle
                        label="Appointment Reminders"
                        value={isRemindersEnabled && isPushEnabled} // Controlled by master switch
                        onValueChange={setIsRemindersEnabled}
                    />
                    <NotificationToggle
                        label="Promotions and Offers"
                        value={isPromosEnabled && isPushEnabled} // Controlled by master switch
                        onValueChange={setIsPromosEnabled}
                    />

                    {/* Example of a setting without a bottom border */}
                    <View style={styles.settingItem}>
                        <Text style={styles.settingText}>Show Notification Previews</Text>
                        <Switch
                            trackColor={{ false: '#B0DAE6', true: '#005A9C' }}
                            thumbColor={'#f4f3f4'}
                            value={true} // Example of a static setting
                            disabled={true}
                        />
                    </View>
                </View>

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
    infoText: {
        fontSize: 14,
        color: '#005A9C',
        textAlign: 'center',
        opacity: 0.8,
    }
});
