# AURELIX Enterprise AI Vision Platform — Administrator Guide

**Version**: 1.0.0 Commercial Release  

---

## 1. System Administration & Licensing
Platform administrators oversee machine binding, license activation keys, security integrity verification, and storage management.

---

## 2. Licensing Procedure
1. Request the user's **Request Code** generated in License Manager (`[16]`).
2. Run the developer signature generator:
   ```python
   from mhyra.launcher.license import LicenseManager
   lm = LicenseManager()
   activation_json = lm.sign_request_code("<REQUEST_CODE>", tier="ENTERPRISE", valid_days=365)
   print(activation_json)
   ```
3. Save the resulting JSON string into `configs/license.key` on the target machine.

---

## 3. Code Integrity & Security Verification
To verify code hashes and ensure anti-tampering guards are active, select **Security & Diagnostics** (`[14]`) from the main console menu.

---
*(c) 2026 AURELIX Technologies Inc. All Rights Reserved.*
