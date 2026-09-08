# HelioDesk support policy handbook

This handbook is the source document for HelioDesk support retrieval practice. It collects operating policies that support specialists must follow when answering customer questions about exports, account access, refunds, invoices, Enterprise commitments, data sharing, trials, security, and internal support notes. The document is intentionally written as normal policy text, not as pre-made retrieval chunks. The notebook loads this document, splits it with the shared `chunk_text(...)` helper, and indexes the generated chunks in ChromaDB.

Support specialists should answer from the policy evidence that is actually retrieved. If the retrieved context does not contain enough evidence for a decision, the specialist should say what is known, explain what is missing, and route the case to the correct internal owner. Support should not invent exceptions, promise timelines that are not written in policy, or cite internal-only controls in customer-facing messages.

## Customer data exports

Support agents can request a customer data export from the workspace settings panel after confirming that the requester is an account owner or workspace administrator. Export jobs include account profile data, ticket metadata, billing contact records, agent notes, macro usage, and audit-log entries for the selected workspace. Exports do not include another customer's data, deleted message attachments that have already aged out of retention, or employee-only investigation notes.

Before starting an export, support must confirm the workspace slug, requester email, and business reason. If the request comes from a contractor, auditor, reseller, or former employee, support must verify that the current account owner has authorized the request. The internal note should record who requested the export, which workspace was selected, and which verification step was completed.

Only workspace administrators and billing managers can download completed export files. Standard support agents may create the export request for an eligible account, but they must ask an administrator or billing manager to download and share the file. If the customer asks for external delivery, support should use the secure customer portal rather than sending the archive as an email attachment.

Completed customer data export files remain available for 30 days after creation. After the 30-day retention window, the archive expires automatically and cannot be recovered from the completed-export page. Support should explain the time limit clearly and create a new export if the customer still needs the records. If the account has been closed, support must check whether the workspace is still inside the account-deletion hold before promising that a new export can be created.

If an export fails, support should first check whether the workspace is under legal hold, whether the requester still has permission, and whether the export queue shows a retryable processing error. Support may retry a failed export once. A second failure must be routed to Support Operations with the export job ID, workspace slug, requester email, and screenshot of the error state.

## Renewal refunds and service credits

Annual-plan customers may request a refund review within 14 days of renewal. Billing operations must check whether the renewal was accidental, whether the workspace continued using paid features, whether the customer received renewal reminders, and whether a previous exception already exists. Support should not promise a refund before billing operations completes the review.

Refund review is not available when the account has already used the renewal-period service credit. In that case, support should explain that the service credit consumed the renewal exception and route any dispute to billing operations rather than promising money back. The customer-facing answer should be brief and should not imply that another agent can override the policy from chat.

Monthly subscriptions are reviewed month by month. Support can cancel future renewal, but the standard policy does not promise a refund for a partially used monthly period. If the customer reports duplicate billing, fraud, or a payment processor error, support should route the case to billing operations instead of applying the monthly-plan rule directly.

Service credits can be applied only by billing operations after a documented incident, duplicate charge, or approved renewal exception. A support specialist may collect the account ID, invoice number, incident ticket, and customer summary, but they should not create a credit memo or change the ledger themselves. Credits appear on the next open invoice unless billing operations states otherwise.

Chargebacks must move to billing operations immediately. Support should acknowledge the customer, avoid arguing about payment liability in the ticket, and preserve the conversation history. Once a chargeback is open, support should not process a separate refund request until billing operations confirms the payment status.

## Enterprise response commitments

Enterprise customers receive onboarding support, account reviews, advanced reporting, export controls, and a named success contact. These benefits describe account coverage, not ticket urgency. The ticket severity still has to match the incident before a faster response commitment applies.

For Enterprise customers, severity-one incidents and critical support tickets receive a first response within 1 business hour. Standard Enterprise tickets receive a response within 4 business hours. If the customer reports security risk, data loss, or a production outage, support should mark the ticket as critical before applying the Enterprise response-time commitment.

The response commitment measures first human response, not final resolution. Support should avoid promising a fix time unless an incident commander has published a resolution estimate. If the issue affects multiple customers or a shared production system, support should link the ticket to the incident channel and follow the incident communication policy.

Enterprise customers may request a named escalation contact when a ticket has missed a commitment, involves account-wide data loss, or blocks a paid launch. The escalation note should summarize customer impact, current owner, requested next step, and any security or legal risk. Escalation is not appropriate for routine feature requests, contract negotiation, or standard billing questions.

## Account access and SSO

Support agents can reset a user's password from the security panel after verifying identity and confirming that the request came from the account owner or a workspace administrator. The reset link expires after 24 hours. If the requester cannot complete verification, the request must move to the Trust and Safety queue.

When single sign-on is enforced, password reset requests are routed to the customer's identity-provider administrator instead of the standard support queue. Support should not send a local reset link because local passwords are disabled for SSO-managed users. If the customer says their company identity provider controls login, the correct next step is to direct them to the identity-provider administrator.

If a teammate loses access to a two-step verification device, an administrator can open a recovery workflow after identity verification. The requester must confirm backup email access before the recovery workflow begins. Support should not remove two-step verification from an account based only on a chat message.

Account-owner transfer requires stronger verification than ordinary password recovery. Support must confirm the current owner, the requested new owner, the workspace slug, and the reason for transfer. If the current owner is unreachable because of employment change, legal dispute, or company acquisition, the case must move to Trust and Safety with written authorization from the customer's organization.

## Historical invoice corrections

Billing managers can reissue invoice copies from the billing ledger and attach a copy to the next statement. Support agents can request a reissue, but they cannot alter tax details or overwrite invoice history on their own.

Billing address corrections affect future invoices only. If a customer needs a corrected historical invoice, billing operations must void and reissue the invoice after validating the tax profile. Support should not tell the customer that an old bill can be edited directly from the support console.

Tax profile changes require the legal business name, billing address, tax registration number when applicable, and the invoice number affected by the request. If the customer asks to change tax details after a reporting period has closed, support must collect the request and route it to billing operations for review.

Payment method updates are self-serve for account owners and billing managers. Support may provide navigation guidance but must not ask customers to paste card numbers, bank details, or one-time passwords into a ticket. If payment details appear in a conversation, support should redact the sensitive value and follow the security handling policy.

## Data sharing and auditor requests

Export links are for internal review unless the customer has requested external delivery through the secure customer portal. Files that contain personal data must not be shared through public links or copied into ticket comments. Support should use the portal link, expiry controls, and audit note required by the data-sharing workflow.

If an external auditor requests customer data, support must verify written authorization from the account owner and log the request in the audit channel before any file is shared. The authorization note should name the auditor, the requested date range, the business reason for the request, and the person who approved access.

Legal requests, subpoenas, and law-enforcement requests must move to Legal Operations. Support should acknowledge receipt, avoid interpreting the request, and attach the original message or document. No customer data should be shared until Legal Operations records the approved response path.

Internal employees may access customer data only for support, security, billing, or approved product operations. Curiosity access is prohibited. If a support specialist sees unrelated customer data while investigating a ticket, they should close the unrelated record and document only the evidence needed for the active case.

## Security incident handling

If a customer reports suspected account compromise, unusual admin activity, exposed export links, or unauthorized access, support should treat the ticket as a security-sensitive case. The specialist should preserve logs, avoid deleting evidence, and route the case to Trust and Safety when access control or account ownership is disputed.

Support may help a verified administrator revoke sessions, rotate API tokens, and disable shared export links. Support should not attribute blame, confirm attacker identity, or promise that no data was accessed unless Trust and Safety has completed the investigation. Customer-facing messages should separate immediate containment steps from investigation status.

If a support specialist accidentally exposes customer data in a ticket, macro, screenshot, or attachment, they must stop work, notify the support lead, and follow the internal data-exposure playbook. The specialist should not quietly delete the message and continue without reporting it.

## Support macros and grounded replies

Reusable support macros should be concise, specific, and grounded in a cited policy source before an agent sends them to a requester. A macro should not invent an exception that is absent from the policy. If the policy requires escalation, the macro should name the receiving team and avoid promising a decision timeline unless the policy states one.

When a question involves account recovery, refunds, exports, security routing, data sharing, or invoice correction, the agent should include the policy source in the internal note. The customer-facing answer can be shorter, but the internal retrieval trail should show which policy informed the response.

Macro edits must be reviewed by Support Operations when they change eligibility, time limits, payment language, escalation ownership, or security instructions. Minor tone edits can be made by support leads. A macro that refers to an old policy date, deprecated product name, or retired workflow should be archived rather than quietly reused.

## Trial extensions and plan changes

Support may grant one 7-day trial extension when the customer has not used a previous extension and the workspace has fewer than 25 active seats. Larger accounts require approval from sales operations. Support should not split a workspace or remove active seats just to fit the trial-extension threshold.

When a trial converts to a paid annual plan, annual renewal rules apply from the paid renewal date rather than the original trial start date. If the customer asks for a trial extension after conversion, support should treat the request as a paid-plan billing question.

Plan downgrade requests take effect at the next renewal unless billing operations approves an exception. Support should explain feature loss, export recommendations, and account-owner approval requirements before routing a downgrade request. If the workspace has Enterprise commitments, the success contact should be notified before downgrade guidance is sent.

## Deletion, retention, and account closure

Account closure requests must be confirmed by the account owner. Support should tell the customer which data may be deleted, which records may remain for legal or billing retention, and how long the account-deletion hold lasts. A workspace inside the deletion hold may still be eligible for a new data export if the owner is verified and the export policy allows it.

Deletion requests that mention legal dispute, security incident, chargeback, or audit investigation require escalation before the account is closed. Support should not complete a deletion request while a legal hold or active security investigation is attached to the workspace. If the policy evidence is unclear, the safer response is to route the case rather than promise deletion.
