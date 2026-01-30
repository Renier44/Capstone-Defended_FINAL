import React from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    SafeAreaView,
    ScrollView,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

export default function PrivacyPolicy() {
    const router = useRouter();

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()}>
                    <MaterialIcons name="arrow-back-ios" size={24} color="#005A9C" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Privacy Policy</Text>
                <View style={{ width: 24 }} />
            </View>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                <View style={styles.policyContainer}>
                    <Text style={styles.policyTitle}>Our Commitment to Your Privacy</Text>
                    <Text style={styles.policyParagraph}>
                        At Smart Sight, your privacy is a top priority. This policy outlines how we handle the information you provide us through the app. Our goal is to be transparent about what data we collect, how we use it, and how we keep it secure.
                    </Text>

                    <Text style={styles.sectionTitle}>Information We Collect</Text>
                    <Text style={styles.policyParagraph}>
                        We collect personal information that you voluntarily provide to us when you register on the app, update your profile, or use our services. This may include your first name, last name, email address, phone number, and physical address. We also collect the profile image you choose to upload.
                    </Text>

                    <Text style={styles.sectionTitle}>How We Use Your Information</Text>
                    <Text style={styles.policyParagraph}>
                        The information we collect is used to manage your account, provide and improve our services, and communicate with you. Your profile details help us personalize your experience and ensure our services are tailored to your needs.
                    </Text>

                    <Text style={styles.sectionTitle}>Data Protection</Text>
                    <Text style={styles.policyParagraph}>
                        We implement a variety of security measures to maintain the safety of your personal information. Your data is stored securely and accessed only by authorized personnel who are required to keep the information confidential.
                    </Text>
                    
                    <Text style={styles.sectionTitle}>Third-Party Sharing</Text>
                    <Text style={styles.policyParagraph}>
                        We do not sell, trade, or otherwise transfer your personally identifiable information to outside parties. This does not include trusted third parties who assist us in operating our app, conducting our business, or serving our users, so long as those parties agree to keep this information confidential.
                    </Text>

                    <Text style={styles.policyDisclaimer}>
                        Note: This is a simplified privacy policy for demonstration purposes. It is not a legally binding document. For a real application, you should consult with legal counsel to draft a comprehensive and legally compliant privacy policy.
                    </Text>
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
    policyContainer: {
        marginHorizontal: 20,
        backgroundColor: '#C8EAF7',
        borderRadius: 25,
        padding: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 10,
    },
    policyTitle: {
        fontSize: 22,
        fontWeight: '700',
        color: '#005A9C',
        textAlign: 'center',
        marginBottom: 20,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: '#005A9C',
        marginTop: 15,
        marginBottom: 5,
    },
    policyParagraph: {
        fontSize: 14,
        color: '#333',
        lineHeight: 22,
        marginBottom: 10,
    },
    policyDisclaimer: {
        fontSize: 12,
        fontStyle: 'italic',
        color: '#666',
        marginTop: 20,
        textAlign: 'center',
    },
});
